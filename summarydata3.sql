WITH p AS (
    SELECT
        TO_DATE(?, 'DDMMYYYY') AS start_dt,
        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt
    FROM dual
),

range_rows AS (
    SELECT
        ROWIDTOCHAR(ph.ROWID) AS rid,
        ph.MODULE_ID,
        ph.PARAM_NAME,
        ph.CREATE_DTTS,
        ph.VALUE,
        ph.LSL,
        ph.USL
    FROM PARAMETER_HISTORY ph
    CROSS JOIN p
    WHERE ph.CREATE_DTTS >= p.start_dt
      AND ph.CREATE_DTTS < p.end_dt
),

before_range AS (
    SELECT
        rid,
        MODULE_ID,
        PARAM_NAME,
        CREATE_DTTS,
        VALUE,
        LSL,
        USL
    FROM (
        SELECT
            ROWIDTOCHAR(ph.ROWID) AS rid,
            ph.MODULE_ID,
            ph.PARAM_NAME,
            ph.CREATE_DTTS,
            ph.VALUE,
            ph.LSL,
            ph.USL,

            ROW_NUMBER() OVER (
                PARTITION BY ph.MODULE_ID, ph.PARAM_NAME
                ORDER BY
                    ph.CREATE_DTTS DESC,
                    ROWIDTOCHAR(ph.ROWID) DESC
            ) AS rn

        FROM PARAMETER_HISTORY ph
        CROSS JOIN p
        WHERE ph.CREATE_DTTS < p.start_dt
    )
    WHERE rn = 1
),

s AS (
    SELECT * FROM range_rows

    UNION ALL

    SELECT * FROM before_range
),

d AS (
    SELECT
        s.rid,
        s.MODULE_ID,
        s.PARAM_NAME,
        s.CREATE_DTTS,

        s.VALUE,

        LEAD(s.VALUE) OVER (
            PARTITION BY s.MODULE_ID, s.PARAM_NAME
            ORDER BY s.CREATE_DTTS DESC, s.rid DESC
        ) AS PREV_VALUE,

        s.LSL,

        LEAD(s.LSL) OVER (
            PARTITION BY s.MODULE_ID, s.PARAM_NAME
            ORDER BY s.CREATE_DTTS DESC, s.rid DESC
        ) AS PREV_LSL,

        s.USL,

        LEAD(s.USL) OVER (
            PARTITION BY s.MODULE_ID, s.PARAM_NAME
            ORDER BY s.CREATE_DTTS DESC, s.rid DESC
        ) AS PREV_USL,

        LEAD(s.CREATE_DTTS) OVER (
            PARTITION BY s.MODULE_ID, s.PARAM_NAME
            ORDER BY s.CREATE_DTTS DESC, s.rid DESC
        ) AS PREV_CREATE_DTTS

    FROM s
),

c AS (
    SELECT
        d.*,

        CASE
            WHEN d.PREV_VALUE IS NOT NULL
             AND d.VALUE IS NOT NULL
             AND DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1
            THEN 1
            ELSE 0
        END AS VALUE_CHANGED,

        CASE
            WHEN d.PREV_LSL IS NOT NULL
             AND d.LSL IS NOT NULL
             AND DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1
            THEN 1
            ELSE 0
        END AS LSL_CHANGED,

        CASE
            WHEN d.PREV_USL IS NOT NULL
             AND d.USL IS NOT NULL
             AND DECODE(d.PREV_USL, d.USL, 0, 1) = 1
            THEN 1
            ELSE 0
        END AS USL_CHANGED

    FROM d
    CROSS JOIN p
    WHERE d.CREATE_DTTS >= p.start_dt
      AND d.CREATE_DTTS < p.end_dt
      AND d.PREV_CREATE_DTTS IS NOT NULL
),

/* Same rows as main data */
main_data AS (
    SELECT
        MODULE_ID,
        PARAM_NAME,
        PREV_CREATE_DTTS AS PREVIOUS_CHANGE_TIME,
        CREATE_DTTS AS CURRENT_CHANGE_TIME,

        CASE
            WHEN VALUE_CHANGED = 1
            THEN PREV_VALUE
        END AS PREV_VALUE,

        CASE
            WHEN VALUE_CHANGED = 1
            THEN VALUE
        END AS VALUE,

        CASE
            WHEN LSL_CHANGED = 1
            THEN PREV_LSL
        END AS PREV_LSL,

        CASE
            WHEN LSL_CHANGED = 1
            THEN LSL
        END AS LSL,

        CASE
            WHEN USL_CHANGED = 1
            THEN PREV_USL
        END AS PREV_USL,

        CASE
            WHEN USL_CHANGED = 1
            THEN USL
        END AS USL,

        VALUE_CHANGED,
        LSL_CHANGED,
        USL_CHANGED,

        RTRIM(
              CASE
                  WHEN VALUE_CHANGED = 1
                  THEN 'VALUE_CHANGED|'
              END
           || CASE
                  WHEN LSL_CHANGED = 1
                  THEN 'LSL_CHANGED|'
              END
           || CASE
                  WHEN USL_CHANGED = 1
                  THEN 'USL_CHANGED|'
              END,
           '|'
        ) AS CHANGE_TYPE

    FROM c

    WHERE VALUE_CHANGED = 1
       OR LSL_CHANGED = 1
       OR USL_CHANGED = 1
)

SELECT
    MODULE_ID,
    PARAM_NAME,

    TO_CHAR(
        MIN(CURRENT_CHANGE_TIME),
        'DD/MM/YYYY HH24:MI:SS'
    ) AS FIRST_CHANGE_TIME,

    TO_CHAR(
        MAX(CURRENT_CHANGE_TIME),
        'DD/MM/YYYY HH24:MI:SS'
    ) AS LAST_CHANGE_TIME,

    SUM(VALUE_CHANGED) AS VALUE_CHANGE_COUNT,

    SUM(LSL_CHANGED) AS LSL_CHANGE_COUNT,

    SUM(USL_CHANGED) AS USL_CHANGE_COUNT,

    COUNT(*) AS TOTAL_CHANGE_POINTS,

    RTRIM(
          CASE
              WHEN SUM(VALUE_CHANGED) > 0
              THEN 'VALUE_CHANGED|'
          END
       || CASE
              WHEN SUM(LSL_CHANGED) > 0
              THEN 'LSL_CHANGED|'
          END
       || CASE
              WHEN SUM(USL_CHANGED) > 0
              THEN 'USL_CHANGED|'
          END,
       '|'
    ) AS CHANGE_TYPE

FROM main_data

GROUP BY
    MODULE_ID,
    PARAM_NAME

ORDER BY
    MAX(CURRENT_CHANGE_TIME) DESC;
