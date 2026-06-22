WITH p AS (
    SELECT
        TO_DATE(?, 'DDMMYYYY') AS start_dt,
        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt,
        TRIM(?) AS module_id,
        TRIM(?) AS param_name
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
    WHERE ph.MODULE_ID = p.module_id
      AND ph.PARAM_NAME = p.param_name
      AND ph.CREATE_DTTS >= p.start_dt
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
GROUP BY
    MODULE_ID,
    PARAM_NAME;
