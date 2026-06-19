WITH p AS (
    SELECT
        TO_DATE(?, 'DDMMYYYY') AS start_dt,
        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt,
        TRIM(?) AS param_name,
        UPPER(TRIM(?)) AS change_type
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
        ) AS PREV_USL

    FROM PARAMETER_HISTORY ph
    CROSS JOIN p
    WHERE ph.PARAM_NAME = p.param_name
      AND ph.CREATE_DTTS >= p.start_dt - 30
      AND ph.CREATE_DTTS <  p.end_dt
),
x AS (
    SELECT
        d.*,
        p.change_type
    FROM d
    CROSS JOIN p
    WHERE d.CREATE_DTTS >= p.start_dt
      AND d.CREATE_DTTS <  p.end_dt
)
SELECT
    PARAM_NAME,
    CREATE_DTTS,

    CASE
        WHEN change_type = 'VALUE' THEN TO_CHAR(PREV_VALUE)
        WHEN change_type = 'LSL'   THEN TO_CHAR(PREV_LSL)
        WHEN change_type = 'USL'   THEN TO_CHAR(PREV_USL)
    END AS PREVIOUS_VALUE,

    CASE
        WHEN change_type = 'VALUE' THEN TO_CHAR(VALUE)
        WHEN change_type = 'LSL'   THEN TO_CHAR(LSL)
        WHEN change_type = 'USL'   THEN TO_CHAR(USL)
    END AS CURRENT_VALUE,

    change_type,

    COUNT(*) OVER () AS CHANGE_COUNT

FROM x
WHERE
       (change_type = 'VALUE' AND DECODE(PREV_VALUE, VALUE, 0, 1) = 1)
    OR (change_type = 'LSL'   AND DECODE(PREV_LSL, LSL, 0, 1) = 1)
    OR (change_type = 'USL'   AND DECODE(PREV_USL, USL, 0, 1) = 1)
ORDER BY CREATE_DTTS;
