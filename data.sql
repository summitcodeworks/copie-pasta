SELECT
    PRODUCTID,
    MODULE_ID,
    PARAM_NAME,
    create_dtts,

    CASE
        WHEN UPPER(TRIM(:change_type)) = 'VALUE' THEN TO_CHAR(prev_value)
        WHEN UPPER(TRIM(:change_type)) = 'LSL'   THEN TO_CHAR(prev_lsl)
        WHEN UPPER(TRIM(:change_type)) = 'USL'   THEN TO_CHAR(prev_usl)
    END AS previous_value,

    CASE
        WHEN UPPER(TRIM(:change_type)) = 'VALUE' THEN TO_CHAR(value)
        WHEN UPPER(TRIM(:change_type)) = 'LSL'   THEN TO_CHAR(lsl)
        WHEN UPPER(TRIM(:change_type)) = 'USL'   THEN TO_CHAR(usl)
    END AS current_value,

    UPPER(TRIM(:change_type)) AS change_type

FROM (
    SELECT
        ph.PRODUCTID,
        ph.MODULE_ID,
        ph.PARAM_NAME,
        ph.create_dtts,

        ph.value,
        LAG(ph.value) OVER (
            PARTITION BY ph.PRODUCTID, ph.MODULE_ID, ph.PARAM_NAME
            ORDER BY ph.create_dtts, ph.ROWID
        ) AS prev_value,

        ph.lsl,
        LAG(ph.lsl) OVER (
            PARTITION BY ph.PRODUCTID, ph.MODULE_ID, ph.PARAM_NAME
            ORDER BY ph.create_dtts, ph.ROWID
        ) AS prev_lsl,

        ph.usl,
        LAG(ph.usl) OVER (
            PARTITION BY ph.PRODUCTID, ph.MODULE_ID, ph.PARAM_NAME
            ORDER BY ph.create_dtts, ph.ROWID
        ) AS prev_usl

    FROM parameter_history ph
    WHERE ph.PRODUCTID = :productid
      AND ph.MODULE_ID = :module_id
      AND ph.PARAM_NAME = :param_name

      -- Keep little extra old data so first selected date can compare previous value
      AND ph.create_dtts >= TO_DATE(:start_date, 'DDMMYYYY') - 30
      AND ph.create_dtts <  TO_DATE(:end_date, 'DDMMYYYY') + 1
)
WHERE create_dtts >= TO_DATE(:start_date, 'DDMMYYYY')
  AND create_dtts <  TO_DATE(:end_date, 'DDMMYYYY') + 1
  AND (
         (UPPER(TRIM(:change_type)) = 'VALUE' AND DECODE(prev_value, value, 0, 1) = 1)
      OR (UPPER(TRIM(:change_type)) = 'LSL'   AND DECODE(prev_lsl, lsl, 0, 1) = 1)
      OR (UPPER(TRIM(:change_type)) = 'USL'   AND DECODE(prev_usl, usl, 0, 1) = 1)
  )
ORDER BY create_dtts;
