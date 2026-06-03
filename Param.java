import java.sql.*;

public class ParameterChangeReport {

    private static final String URL =
            "jdbc:oracle:thin:@//hostname:1521/service_name";

    private static final String USERNAME = "username";
    private static final String PASSWORD = "password";

    private static final String QUERY = """
        WITH data AS (
            SELECT
                product_id,
                module_id,
                parameter_name,
                create_dtts,

                LAG(create_dtts) OVER (
                    PARTITION BY product_id, module_id, parameter_name
                    ORDER BY create_dtts
                ) last_changed_date,

                value,
                LAG(value) OVER (
                    PARTITION BY product_id, module_id, parameter_name
                    ORDER BY create_dtts
                ) prev_value,

                lsl,
                LAG(lsl) OVER (
                    PARTITION BY product_id, module_id, parameter_name
                    ORDER BY create_dtts
                ) prev_lsl,

                usl,
                LAG(usl) OVER (
                    PARTITION BY product_id, module_id, parameter_name
                    ORDER BY create_dtts
                ) prev_usl
            FROM parameter_history
            WHERE create_dtts >= SYSDATE - 7
        ),
        changes AS (
            SELECT
                d.*,
                CASE
                    WHEN NVL(prev_value,'~') <> NVL(value,'~')
                        THEN 'VALUE_CHANGED'
                    WHEN NVL(prev_lsl,-999999999) <> NVL(lsl,-999999999)
                        THEN 'LSL_CHANGED'
                    WHEN NVL(prev_usl,-999999999) <> NVL(usl,-999999999)
                        THEN 'USL_CHANGED'
                END change_type,
                ROW_NUMBER() OVER (
                    PARTITION BY product_id, module_id, parameter_name
                    ORDER BY create_dtts DESC
                ) rn
            FROM data d
            WHERE
                   NVL(prev_value,'~') <> NVL(value,'~')
                OR NVL(prev_lsl,-999999999) <> NVL(lsl,-999999999)
                OR NVL(prev_usl,-999999999) <> NVL(usl,-999999999)
        )
        SELECT
            product_id,
            module_id,
            parameter_name,
            last_changed_date,
            create_dtts AS current_changed_date,
            prev_value,
            value,
            prev_lsl,
            lsl,
            prev_usl,
            usl,
            change_type
        FROM changes
        WHERE rn = 1
        ORDER BY product_id,
                 module_id,
                 parameter_name
        """;

    public static void main(String[] args) {

        try (Connection conn =
                     DriverManager.getConnection(URL, USERNAME, PASSWORD);
             PreparedStatement ps =
                     conn.prepareStatement(QUERY);
             ResultSet rs = ps.executeQuery()) {

            while (rs.next()) {

                String productId = rs.getString("PRODUCT_ID");
                String moduleId = rs.getString("MODULE_ID");
                String parameterName = rs.getString("PARAMETER_NAME");

                Timestamp lastChangedDate =
                        rs.getTimestamp("LAST_CHANGED_DATE");

                Timestamp currentChangedDate =
                        rs.getTimestamp("CURRENT_CHANGED_DATE");

                String prevValue = rs.getString("PREV_VALUE");
                String currentValue = rs.getString("VALUE");

                String prevLsl = rs.getString("PREV_LSL");
                String currentLsl = rs.getString("LSL");

                String prevUsl = rs.getString("PREV_USL");
                String currentUsl = rs.getString("USL");

                String changeType =
                        rs.getString("CHANGE_TYPE");

                System.out.println(
                        productId + "," +
                        moduleId + "," +
                        parameterName + "," +
                        lastChangedDate + "," +
                        currentChangedDate + "," +
                        prevValue + "," +
                        currentValue + "," +
                        prevLsl + "," +
                        currentLsl + "," +
                        prevUsl + "," +
                        currentUsl + "," +
                        changeType
                );
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
