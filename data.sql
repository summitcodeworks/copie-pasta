WITH d AS (
    SELECT
        PARAM_NAME,
        create_dtts,

        value,
        LAG(value) OVER (
            PARTITION BY PARAM_NAME
            ORDER BY create_dtts, ROWID
        ) AS prev_value,

        lsl,
        LAG(lsl) OVER (
            PARTITION BY PARAM_NAME
            ORDER BY create_dtts, ROWID
        ) AS prev_lsl,

        usl,
        LAG(usl) OVER (
            PARTITION BY PARAM_NAME
            ORDER BY create_dtts, ROWID
        ) AS prev_usl

    FROM parameter_history
    WHERE PARAM_NAME = :param_name
      AND create_dtts >= TO_DATE(:start_date, 'DDMMYYYY') - 1
      AND create_dtts <  TO_DATE(:end_date, 'DDMMYYYY') + 1
),
x AS (
    SELECT
        PARAM_NAME,
        create_dtts,

        CASE
            WHEN UPPER(:change_type) IN ('VALUE', 'VALUE_CHANGED') THEN TO_CHAR(prev_value)
            WHEN UPPER(:change_type) IN ('LSL', 'LSL_CHANGED')     THEN TO_CHAR(prev_lsl)
            WHEN UPPER(:change_type) IN ('USL', 'USL_CHANGED')     THEN TO_CHAR(prev_usl)
        END AS previous_value,

        CASE
            WHEN UPPER(:change_type) IN ('VALUE', 'VALUE_CHANGED') THEN TO_CHAR(value)
            WHEN UPPER(:change_type) IN ('LSL', 'LSL_CHANGED')     THEN TO_CHAR(lsl)
            WHEN UPPER(:change_type) IN ('USL', 'USL_CHANGED')     THEN TO_CHAR(usl)
        END AS current_value,

        CASE
            WHEN UPPER(:change_type) IN ('VALUE', 'VALUE_CHANGED')
             AND DECODE(prev_value, value, 0, 1) = 1 THEN 1

            WHEN UPPER(:change_type) IN ('LSL', 'LSL_CHANGED')
             AND DECODE(prev_lsl, lsl, 0, 1) = 1 THEN 1

            WHEN UPPER(:change_type) IN ('USL', 'USL_CHANGED')
             AND DECODE(prev_usl, usl, 0, 1) = 1 THEN 1

            ELSE 0
        END AS is_changed

    FROM d
    WHERE create_dtts >= TO_DATE(:start_date, 'DDMMYYYY')
      AND create_dtts <  TO_DATE(:end_date, 'DDMMYYYY') + 1
)
SELECT
    PARAM_NAME,
    create_dtts,
    previous_value,
    current_value,
    current_value AS graph_value,
    COUNT(*) OVER () AS change_count
FROM x
WHERE is_changed = 1
ORDER BY create_dtts;
