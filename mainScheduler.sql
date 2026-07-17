-- mainScheduler.sql
-- Oracle DBMS_SCHEDULER deployment script.
--
-- Creates:
--   1. Procedure : PRC_CAPTURE_PARAMETER_CHANGES
--   2. Schedule  : SCH_PARAMETER_CHANGES_HOURLY
--   3. Program   : PRG_CAPTURE_PARAMETER_CHANGES
--   4. Job       : JOB_CAPTURE_PARAMETER_CHANGES
--
-- The scheduler owner needs CREATE JOB and permission to read
-- PARAMETER_HISTORY and write to PARAMETER_CHANGE_AUDIT.

SET DEFINE OFF;
SET SERVEROUTPUT ON;

--------------------------------------------------------------------------------
-- Supporting result table
--------------------------------------------------------------------------------
-- Adjust VARCHAR2 lengths below if the source columns are larger in your schema.
BEGIN
    EXECUTE IMMEDIATE q'[
        CREATE TABLE PARAMETER_CHANGE_AUDIT (
            MODULE_ID             VARCHAR2(200)  NOT NULL,
            PARAM_NAME            VARCHAR2(200)  NOT NULL,
            PREVIOUS_CHANGE_TIME  DATE           NOT NULL,
            CURRENT_CHANGE_TIME   DATE           NOT NULL,
            PREV_VALUE            VARCHAR2(4000),
            VALUE                 VARCHAR2(4000),
            PREV_LSL              VARCHAR2(4000),
            LSL                   VARCHAR2(4000),
            PREV_USL              VARCHAR2(4000),
            USL                   VARCHAR2(4000),
            CHANGE_TYPE           VARCHAR2(100)  NOT NULL,
            CAPTURED_AT           TIMESTAMP(6) DEFAULT SYSTIMESTAMP NOT NULL,
            CONSTRAINT PK_PARAMETER_CHANGE_AUDIT PRIMARY KEY
                (MODULE_ID, PARAM_NAME, CURRENT_CHANGE_TIME)
        )
    ]';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN -- ORA-00955: object already exists
            RAISE;
        END IF;
END;
/

--------------------------------------------------------------------------------
-- 1. Procedure
--------------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE PRC_CAPTURE_PARAMETER_CHANGES
AUTHID DEFINER
AS
BEGIN
    MERGE INTO PARAMETER_CHANGE_AUDIT target
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
                d.*,
                CASE WHEN DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1
                     THEN 1 ELSE 0 END AS VALUE_CHANGED,
                CASE WHEN DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1
                     THEN 1 ELSE 0 END AS LSL_CHANGED,
                CASE WHEN DECODE(d.PREV_USL, d.USL, 0, 1) = 1
                     THEN 1 ELSE 0 END AS USL_CHANGED
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
            MODULE_ID,
            PARAM_NAME,
            PREV_CREATE_DTTS AS PREVIOUS_CHANGE_TIME,
            CREATE_DTTS      AS CURRENT_CHANGE_TIME,
            CASE WHEN VALUE_CHANGED = 1 THEN TO_CHAR(PREV_VALUE) END AS PREV_VALUE,
            CASE WHEN VALUE_CHANGED = 1 THEN TO_CHAR(VALUE) END      AS VALUE,
            CASE WHEN LSL_CHANGED = 1 THEN TO_CHAR(PREV_LSL) END     AS PREV_LSL,
            CASE WHEN LSL_CHANGED = 1 THEN TO_CHAR(LSL) END          AS LSL,
            CASE WHEN USL_CHANGED = 1 THEN TO_CHAR(PREV_USL) END     AS PREV_USL,
            CASE WHEN USL_CHANGED = 1 THEN TO_CHAR(USL) END          AS USL,
            RTRIM(
                  CASE WHEN VALUE_CHANGED = 1 THEN 'VALUE_CHANGED|' END
               || CASE WHEN LSL_CHANGED = 1 THEN 'LSL_CHANGED|' END
               || CASE WHEN USL_CHANGED = 1 THEN 'USL_CHANGED|' END,
               '|'
            ) AS CHANGE_TYPE
        FROM c
    ) source
    ON (
           target.MODULE_ID = source.MODULE_ID
       AND target.PARAM_NAME = source.PARAM_NAME
       AND target.CURRENT_CHANGE_TIME = source.CURRENT_CHANGE_TIME
    )
    WHEN MATCHED THEN UPDATE SET
        target.PREVIOUS_CHANGE_TIME = source.PREVIOUS_CHANGE_TIME,
        target.PREV_VALUE           = source.PREV_VALUE,
        target.VALUE                = source.VALUE,
        target.PREV_LSL             = source.PREV_LSL,
        target.LSL                  = source.LSL,
        target.PREV_USL             = source.PREV_USL,
        target.USL                  = source.USL,
        target.CHANGE_TYPE          = source.CHANGE_TYPE,
        target.CAPTURED_AT          = SYSTIMESTAMP
    WHEN NOT MATCHED THEN INSERT (
        MODULE_ID, PARAM_NAME, PREVIOUS_CHANGE_TIME, CURRENT_CHANGE_TIME,
        PREV_VALUE, VALUE, PREV_LSL, LSL, PREV_USL, USL,
        CHANGE_TYPE, CAPTURED_AT
    ) VALUES (
        source.MODULE_ID, source.PARAM_NAME,
        source.PREVIOUS_CHANGE_TIME, source.CURRENT_CHANGE_TIME,
        source.PREV_VALUE, source.VALUE, source.PREV_LSL, source.LSL,
        source.PREV_USL, source.USL, source.CHANGE_TYPE, SYSTIMESTAMP
    );

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END PRC_CAPTURE_PARAMETER_CHANGES;
/

--------------------------------------------------------------------------------
-- Remove prior Scheduler objects so the script can be run again safely.
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.DROP_JOB(
        job_name => 'JOB_CAPTURE_PARAMETER_CHANGES',
        force    => TRUE
    );
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27475 THEN RAISE; END IF;
END;
/

BEGIN
    DBMS_SCHEDULER.DROP_PROGRAM(
        program_name => 'PRG_CAPTURE_PARAMETER_CHANGES',
        force        => TRUE
    );
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27476 THEN RAISE; END IF;
END;
/

BEGIN
    DBMS_SCHEDULER.DROP_SCHEDULE(
        schedule_name => 'SCH_PARAMETER_CHANGES_HOURLY',
        force         => TRUE
    );
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -27476 THEN RAISE; END IF;
END;
/

--------------------------------------------------------------------------------
-- 2. Named hourly schedule
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.CREATE_SCHEDULE(
        schedule_name   => 'SCH_PARAMETER_CHANGES_HOURLY',
        start_date      => SYSTIMESTAMP,
        repeat_interval => 'FREQ=HOURLY;INTERVAL=1',
        comments        => 'Runs once every hour.'
    );
END;
/

--------------------------------------------------------------------------------
-- 3. Program that invokes the procedure
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.CREATE_PROGRAM(
        program_name        => 'PRG_CAPTURE_PARAMETER_CHANGES',
        program_type        => 'STORED_PROCEDURE',
        program_action      => 'PRC_CAPTURE_PARAMETER_CHANGES',
        number_of_arguments => 0,
        enabled             => TRUE,
        comments            => 'Captures changes from PARAMETER_HISTORY.'
    );
END;
/

--------------------------------------------------------------------------------
-- 4. Job connecting the program to the hourly schedule
--------------------------------------------------------------------------------
BEGIN
    DBMS_SCHEDULER.CREATE_JOB(
        job_name      => 'JOB_CAPTURE_PARAMETER_CHANGES',
        program_name  => 'PRG_CAPTURE_PARAMETER_CHANGES',
        schedule_name => 'SCH_PARAMETER_CHANGES_HOURLY',
        enabled       => TRUE,
        auto_drop     => FALSE,
        comments      => 'Hourly parameter-change capture job.'
    );

    DBMS_SCHEDULER.SET_ATTRIBUTE(
        name      => 'JOB_CAPTURE_PARAMETER_CHANGES',
        attribute => 'store_output',
        value     => TRUE
    );
END;
/

--------------------------------------------------------------------------------
-- Verification queries
--------------------------------------------------------------------------------
SELECT object_name, object_type, status
FROM user_objects
WHERE object_name IN (
    'PRC_CAPTURE_PARAMETER_CHANGES',
    'PARAMETER_CHANGE_AUDIT'
)
ORDER BY object_type, object_name;

SELECT job_name, enabled, state, repeat_interval, last_start_date, next_run_date
FROM user_scheduler_jobs
WHERE job_name = 'JOB_CAPTURE_PARAMETER_CHANGES';

-- Optional immediate test:
-- BEGIN
--     DBMS_SCHEDULER.RUN_JOB('JOB_CAPTURE_PARAMETER_CHANGES', use_current_session => TRUE);
-- END;
-- /

