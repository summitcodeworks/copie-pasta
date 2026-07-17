-- summaryScheduler.sql
-- Hourly Oracle Scheduler deployment for parameter-change summaries.

SET DEFINE OFF;
SET SERVEROUTPUT ON;

--------------------------------------------------------------------------------
-- Supporting summary table
--------------------------------------------------------------------------------
BEGIN
    EXECUTE IMMEDIATE q'[
        CREATE TABLE PARAMETER_CHANGE_SUMMARY (
            SUMMARY_DATE         DATE          NOT NULL,
            MODULE_ID            VARCHAR2(200) NOT NULL,
            PARAM_NAME           VARCHAR2(200) NOT NULL,
            FIRST_CHANGE_TIME    DATE          NOT NULL,
            LAST_CHANGE_TIME     DATE          NOT NULL,
            VALUE_CHANGE_COUNT   NUMBER        NOT NULL,
            LSL_CHANGE_COUNT     NUMBER        NOT NULL,
            USL_CHANGE_COUNT     NUMBER        NOT NULL,
            TOTAL_CHANGE_POINTS  NUMBER        NOT NULL,
            CHANGE_TYPE          VARCHAR2(100) NOT NULL,
            CAPTURED_AT          TIMESTAMP(6) DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT PK_PARAMETER_CHANGE_SUMMARY PRIMARY KEY
                (SUMMARY_DATE, MODULE_ID, PARAM_NAME)
        )
    ]';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;
/

--------------------------------------------------------------------------------
-- 1. Procedure
--------------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE PRC_CAPTURE_CHANGE_SUMMARY
AUTHID DEFINER
AS
BEGIN
    MERGE INTO PARAMETER_CHANGE_SUMMARY target
    USING (
        WITH p AS (
            SELECT
                TRUNC(SYSDATE)     AS start_dt,
                TRUNC(SYSDATE) + 1 AS end_dt
            FROM dual
        ),
        d AS (
            SELECT
                ph.ROWID AS rid,
                ph.MODULE_ID,
                ph.PARAM_NAME,
                ph.CREATE_DTTS,
                ph.VALUE,
                LEAD(ph.VALUE) OVER (
                    PARTITION BY ph.MODULE_ID, ph.PARAM_NAME
                    ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC
                ) AS PREV_VALUE,
                ph.LSL,
                LEAD(ph.LSL) OVER (
                    PARTITION BY ph.MODULE_ID, ph.PARAM_NAME
                    ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC
                ) AS PREV_LSL,
                ph.USL,
                LEAD(ph.USL) OVER (
                    PARTITION BY ph.MODULE_ID, ph.PARAM_NAME
                    ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC
                ) AS PREV_USL,
                LEAD(ph.CREATE_DTTS) OVER (
                    PARTITION BY ph.MODULE_ID, ph.PARAM_NAME
                    ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC
                ) AS PREV_CREATE_DTTS
            FROM PARAMETER_HISTORY ph
            CROSS JOIN p
            WHERE ph.CREATE_DTTS >= p.start_dt
              AND ph.CREATE_DTTS <  p.end_dt
        ),
        c AS (
            SELECT
                d.MODULE_ID,
                d.PARAM_NAME,
                d.CREATE_DTTS,
                CASE
                    WHEN DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1
                    THEN 1 ELSE 0
                END AS VALUE_CHANGED,
                CASE
                    WHEN DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1
                    THEN 1 ELSE 0
                END AS LSL_CHANGED,
                CASE
                    WHEN DECODE(d.PREV_USL, d.USL, 0, 1) = 1
                    THEN 1 ELSE 0
                END AS USL_CHANGED
            FROM d
            WHERE d.PREV_CREATE_DTTS IS NOT NULL
              AND d.PREV_VALUE IS NOT NULL
              AND d.PREV_LSL IS NOT NULL
              AND d.PREV_USL IS NOT NULL
              AND d.VALUE IS NOT NULL
              AND d.LSL IS NOT NULL
              AND d.USL IS NOT NULL
              AND (
                     DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1
                  OR DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1
                  OR DECODE(d.PREV_USL, d.USL, 0, 1) = 1
              )
        )
        SELECT
            TRUNC(SYSDATE) AS SUMMARY_DATE,
            MODULE_ID,
            PARAM_NAME,
            MIN(CREATE_DTTS) AS FIRST_CHANGE_TIME,
            MAX(CREATE_DTTS) AS LAST_CHANGE_TIME,
            SUM(VALUE_CHANGED) AS VALUE_CHANGE_COUNT,
            SUM(LSL_CHANGED) AS LSL_CHANGE_COUNT,
            SUM(USL_CHANGED) AS USL_CHANGE_COUNT,
            COUNT(*) AS TOTAL_CHANGE_POINTS,
            RTRIM(
                  CASE WHEN SUM(VALUE_CHANGED) > 0 THEN 'VALUE_CHANGED|' END
               || CASE WHEN SUM(LSL_CHANGED) > 0 THEN 'LSL_CHANGED|' END
               || CASE WHEN SUM(USL_CHANGED) > 0 THEN 'USL_CHANGED|' END,
               '|'
            ) AS CHANGE_TYPE
        FROM c
        GROUP BY MODULE_ID, PARAM_NAME
    ) source
    ON (
           target.SUMMARY_DATE = source.SUMMARY_DATE
       AND target.MODULE_ID = source.MODULE_ID
       AND target.PARAM_NAME = source.PARAM_NAME
    )
    WHEN MATCHED THEN UPDATE SET
        target.FIRST_CHANGE_TIME   = source.FIRST_CHANGE_TIME,
        target.LAST_CHANGE_TIME    = source.LAST_CHANGE_TIME,
        target.VALUE_CHANGE_COUNT  = source.VALUE_CHANGE_COUNT,
        target.LSL_CHANGE_COUNT    = source.LSL_CHANGE_COUNT,
        target.USL_CHANGE_COUNT    = source.USL_CHANGE_COUNT,
        target.TOTAL_CHANGE_POINTS = source.TOTAL_CHANGE_POINTS,
        target.CHANGE_TYPE         = source.CHANGE_TYPE,
        target.CAPTURED_AT         = SYSTIMESTAMP
    WHEN NOT MATCHED THEN INSERT (
        SUMMARY_DATE, MODULE_ID, PARAM_NAME,
        FIRST_CHANGE_TIME, LAST_CHANGE_TIME,
        VALUE_CHANGE_COUNT, LSL_CHANGE_COUNT, USL_CHANGE_COUNT,
        TOTAL_CHANGE_POINTS, CHANGE_TYPE, CAPTURED_AT
    ) VALUES (
        source.SUMMARY_DATE, source.MODULE_ID, source.PARAM_NAME,
        source.FIRST_CHANGE_TIME, source.LAST_CHANGE_TIME,
        source.VALUE_CHANGE_COUNT, source.LSL_CHANGE_COUNT,
        source.USL_CHANGE_COUNT, source.TOTAL_CHANGE_POINTS,
        source.CHANGE_TYPE, SYSTIMESTAMP
    );

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END PRC_CAPTURE_CHANGE_SUMMARY;
/

--------------------------------------------------------------------------------
-- Drop old Scheduler objects, allowing this script to be rerun.
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.DROP_JOB('JOB_CAPTURE_CHANGE_SUMMARY', force => TRUE);
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27475 THEN RAISE; END IF;
END;
/

BEGIN
    DBMS_SCHEDULER.DROP_PROGRAM('PRG_CAPTURE_CHANGE_SUMMARY', force => TRUE);
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27476 THEN RAISE; END IF;
END;
/

BEGIN
    DBMS_SCHEDULER.DROP_SCHEDULE('SCH_CHANGE_SUMMARY_HOURLY', force => TRUE);
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27476 THEN RAISE; END IF;
END;
/

--------------------------------------------------------------------------------
-- 2. Hourly schedule
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.CREATE_SCHEDULE(
        schedule_name   => 'SCH_CHANGE_SUMMARY_HOURLY',
        start_date      => SYSTIMESTAMP,
        repeat_interval => 'FREQ=HOURLY;INTERVAL=1',
        comments        => 'Runs the parameter-change summary every hour.'
    );
END;
/

--------------------------------------------------------------------------------
-- 3. Program
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.CREATE_PROGRAM(
        program_name        => 'PRG_CAPTURE_CHANGE_SUMMARY',
        program_type        => 'STORED_PROCEDURE',
        program_action      => 'PRC_CAPTURE_CHANGE_SUMMARY',
        number_of_arguments => 0,
        enabled             => TRUE,
        comments            => 'Creates the daily parameter-change summary.'
    );
END;
/

--------------------------------------------------------------------------------
-- 4. Job
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.CREATE_JOB(
        job_name      => 'JOB_CAPTURE_CHANGE_SUMMARY',
        program_name  => 'PRG_CAPTURE_CHANGE_SUMMARY',
        schedule_name => 'SCH_CHANGE_SUMMARY_HOURLY',
        enabled       => TRUE,
        auto_drop     => FALSE,
        comments      => 'Hourly parameter-change summary job.'
    );

    DBMS_SCHEDULER.SET_ATTRIBUTE(
        name      => 'JOB_CAPTURE_CHANGE_SUMMARY',
        attribute => 'store_output',
        value     => TRUE
    );
END;
/

--------------------------------------------------------------------------------
-- Verification
--------------------------------------------------------------------------------
SELECT object_name, object_type, status
FROM user_objects
WHERE object_name IN ('PRC_CAPTURE_CHANGE_SUMMARY', 'PARAMETER_CHANGE_SUMMARY')
ORDER BY object_type, object_name;

SELECT job_name, enabled, state, repeat_interval, last_start_date, next_run_date
FROM user_scheduler_jobs
WHERE job_name = 'JOB_CAPTURE_CHANGE_SUMMARY';

-- Optional immediate test:
-- BEGIN
--     DBMS_SCHEDULER.RUN_JOB('JOB_CAPTURE_CHANGE_SUMMARY',
--                            use_current_session => TRUE);
-- END;
-- /
