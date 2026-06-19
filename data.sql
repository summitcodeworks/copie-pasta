WITH params AS (
    SELECT
        TO_DATE(:start_date, 'DDMMYYYY') AS start_dt,
        TO_DATE(:end_date, 'DDMMYYYY') + 1 AS end_dt,
        :parameter_name AS parameter_name,
        UPPER(:change_type) AS change_type
    FROM dual
),
range_rows AS (
    SELECT
        ph.ROWID AS rid,
        ph.product_id,
        ph.module_id,
        ph.parameter_name,
        ph.create_dtts,
        ph.value,
        ph.lsl,
        ph.usl,
        1 AS is_range_row
    FROM parameter_history ph
    CROSS JOIN params p
    WHERE ph.parameter_name = p.parameter_name
      AND ph.create_dtts >= p.start_dt
      AND ph.create_dtts <  p.end_dt
),
range_keys AS (
    SELECT DISTINCT
        product_id,
        module_id,
        parameter_name
    FROM range_rows
),
prev_rows AS (
    SELECT
        rid,
        product_id,
        module_id,
        parameter_name,
        create_dtts,
        value,
        lsl,
        usl,
        0 AS is_range_row
    FROM (
        SELECT
            ph.ROWID AS rid,
            ph.product_id,
            ph.module_id,
            ph.parameter_name,
            ph.create_dtts,
            ph.value,
            ph.lsl,
            ph.usl,

            ROW_NUMBER() OVER (
                PARTITION BY ph.product_id, ph.module_id, ph.parameter_name
                ORDER BY ph.create_dtts DESC, ph.ROWID DESC
            ) AS rn

        FROM parameter_history ph
        JOIN range_keys rk
          ON rk.product_id = ph.product_id
         AND rk.module_id = ph.module_id
         AND rk.parameter_name = ph.parameter_name
        CROSS JOIN params p
        WHERE ph.create_dtts < p.start_dt
    )
    WHERE rn = 1
),
all_rows AS (
    SELECT * FROM prev_rows
    UNION ALL
    SELECT * FROM range_rows
),
data AS (
    SELECT
        product_id,
        module_id,
        parameter_name,
        create_dtts,
        is_range_row,

        value,
        LAG(value) OVER (
            PARTITION BY product_id, module_id, parameter_name
            ORDER BY create_dtts, rid
        ) AS prev_value,

        lsl,
        LAG(lsl) OVER (
            PARTITION BY product_id, module_id, parameter_name
            ORDER BY create_dtts, rid
        ) AS prev_lsl,

        usl,
        LAG(usl) OVER (
            PARTITION BY product_id, module_id, parameter_name
            ORDER BY create_dtts, rid
        ) AS prev_usl

    FROM all_rows
),
flags AS (
    SELECT
        d.product_id,
        d.module_id,
        d.parameter_name,
        d.create_dtts,

        d.prev_value,
        d.value,

        d.prev_lsl,
        d.lsl,

        d.prev_usl,
        d.usl,

        CASE
            WHEN DECODE(d.prev_value, d.value, 0, 1) = 1
            THEN 1 ELSE 0
        END AS value_change_flag,

        CASE
            WHEN DECODE(d.prev_lsl, d.lsl, 0, 1) = 1
            THEN 1 ELSE 0
        END AS lsl_change_flag,

        CASE
            WHEN DECODE(d.prev_usl, d.usl, 0, 1) = 1
            THEN 1 ELSE 0
        END AS usl_change_flag

    FROM data d
    WHERE d.is_range_row = 1
),
filtered AS (
    SELECT
        f.*,

        CASE
            WHEN p.change_type = 'VALUE' THEN TO_CHAR(f.prev_value)
            WHEN p.change_type = 'LSL'   THEN TO_CHAR(f.prev_lsl)
            WHEN p.change_type = 'USL'   THEN TO_CHAR(f.prev_usl)
        END AS previous_selected_value,

        CASE
            WHEN p.change_type = 'VALUE' THEN TO_CHAR(f.value)
            WHEN p.change_type = 'LSL'   THEN TO_CHAR(f.lsl)
            WHEN p.change_type = 'USL'   THEN TO_CHAR(f.usl)
        END AS current_selected_value,

        p.change_type

    FROM flags f
    CROSS JOIN params p
    WHERE
           (p.change_type = 'VALUE' AND f.value_change_flag = 1)
        OR (p.change_type = 'LSL'   AND f.lsl_change_flag = 1)
        OR (p.change_type = 'USL'   AND f.usl_change_flag = 1)
)
SELECT
    product_id,
    module_id,
    parameter_name,

    create_dtts,

    change_type,

    previous_selected_value,
    current_selected_value,

    COUNT(*) OVER (
        PARTITION BY product_id, module_id, parameter_name, change_type
    ) AS selected_change_count

FROM filtered
ORDER BY
    product_id,
    module_id,
    parameter_name,
    create_dtts;
