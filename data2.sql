WITH p AS (
    SELECT
        TO_DATE(?, 'DDMMYYYY') AS start_dt,
        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt,
        TRIM(?) AS param_name
    FROM dual
),
d AS (
    SELECT
        ph.PARAM_NAME,
        ph.CREATE_DTTS,

        ph.VALUE,
        LAG(ph.VALUE) OVER (
            PARTITION BY ph.PARAM_NAME
            ORDER BY ph.CREATE_DTTS, ph.ROWID
        ) AS PREV_VALUE,

        ph.LSL,
        LAG(ph.LSL) OVER (
            PARTITION BY ph.PARAM_NAME
            ORDER BY ph.CREATE_DTTS, ph.ROWID
        ) AS PREV_LSL,

        ph.USL,
        LAG(ph.USL) OVER (
            PARTITION BY ph.PARAM_NAME
            ORDER BY ph.CREATE_DTTS, ph.ROWID
        ) AS PREV_USL,

        LAG(ph.CREATE_DTTS) OVER (
            PARTITION BY ph.PARAM_NAME
            ORDER BY ph.CREATE_DTTS, ph.ROWID
        ) AS PREV_CREATE_DTTS

    FROM PARAMETER_HISTORY ph
    CROSS JOIN p
    WHERE ph.PARAM_NAME = p.param_name
      AND ph.CREATE_DTTS >= p.start_dt - 30
      AND ph.CREATE_DTTS <  p.end_dt
),
f AS (
    SELECT
        PARAM_NAME,
        CREATE_DTTS,

        CASE
            WHEN PREV_CREATE_DTTS IS NOT NULL
             AND NVL(TO_CHAR(PREV_VALUE), '~') <> NVL(TO_CHAR(VALUE), '~')
            THEN 1 ELSE 0
        END AS VALUE_CHANGE_FLAG,

        CASE
            WHEN PREV_CREATE_DTTS IS NOT NULL
             AND NVL(TO_CHAR(PREV_LSL), '~') <> NVL(TO_CHAR(LSL), '~')
            THEN 1 ELSE 0
        END AS LSL_CHANGE_FLAG,

        CASE
            WHEN PREV_CREATE_DTTS IS NOT NULL
             AND NVL(TO_CHAR(PREV_USL), '~') <> NVL(TO_CHAR(USL), '~')
            THEN 1 ELSE 0
        END AS USL_CHANGE_FLAG

    FROM d
    CROSS JOIN p
    WHERE CREATE_DTTS >= p.start_dt
      AND CREATE_DTTS <  p.end_dt
)
SELECT
    PARAM_NAME,

    MIN(CASE
            WHEN VALUE_CHANGE_FLAG = 1
              OR LSL_CHANGE_FLAG = 1
              OR USL_CHANGE_FLAG = 1
            THEN CREATE_DTTS
        END) AS FIRST_CHANGE_DTTS,

    MAX(CASE
            WHEN VALUE_CHANGE_FLAG = 1
              OR LSL_CHANGE_FLAG = 1
              OR USL_CHANGE_FLAG = 1
            THEN CREATE_DTTS
        END) AS LAST_CHANGE_DTTS,

    SUM(VALUE_CHANGE_FLAG) AS VALUE_CHANGE_COUNT,
    SUM(LSL_CHANGE_FLAG) AS LSL_CHANGE_COUNT,
    SUM(USL_CHANGE_FLAG) AS USL_CHANGE_COUNT,

    SUM(
        CASE
            WHEN VALUE_CHANGE_FLAG = 1
              OR LSL_CHANGE_FLAG = 1
              OR USL_CHANGE_FLAG = 1
            THEN 1 ELSE 0
        END
    ) AS TOTAL_CHANGE_POINTS

FROM f
GROUP BY PARAM_NAME
HAVING
       SUM(VALUE_CHANGE_FLAG) > 0
    OR SUM(LSL_CHANGE_FLAG) > 0
    OR SUM(USL_CHANGE_FLAG) > 0
ORDER BY PARAM_NAME;
