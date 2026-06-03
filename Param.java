import com.fasterxml.jackson.databind.ObjectMapper;

import javax.servlet.annotation.WebServlet;
import javax.servlet.http.*;
import java.io.IOException;
import java.sql.*;
import java.util.*;

@WebServlet("/parameter-changes")
public class ParameterChangeServlet extends HttpServlet {

    private static final String URL =
            "jdbc:oracle:thin:@//hostname:1521/service_name";

    private static final String USERNAME = "username";
    private static final String PASSWORD = "password";

    private static final String SQL = """
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
            create_dtts current_changed_date,
            prev_value,
            value,
            prev_lsl,
            lsl,
            prev_usl,
            usl,
            change_type
        FROM changes
        WHERE rn = 1
        ORDER BY product_id,module_id,parameter_name
        """;

    @Override
    protected void doGet(
            HttpServletRequest request,
            HttpServletResponse response)
            throws IOException {

        response.setContentType("application/json");

        List<Map<String, Object>> result = new ArrayList<>();

        try (
                Connection conn =
                        DriverManager.getConnection(
                                URL,
                                USERNAME,
                                PASSWORD);

                PreparedStatement ps =
                        conn.prepareStatement(SQL);

                ResultSet rs =
                        ps.executeQuery()
        ) {

            while (rs.next()) {

                Map<String, Object> row = new LinkedHashMap<>();

                row.put("productId",
                        rs.getString("PRODUCT_ID"));

                row.put("moduleId",
                        rs.getString("MODULE_ID"));

                row.put("parameterName",
                        rs.getString("PARAMETER_NAME"));

                row.put("lastChangedDate",
                        rs.getTimestamp("LAST_CHANGED_DATE"));

                row.put("currentChangedDate",
                        rs.getTimestamp("CURRENT_CHANGED_DATE"));

                row.put("previousValue",
                        rs.getString("PREV_VALUE"));

                row.put("currentValue",
                        rs.getString("VALUE"));

                row.put("previousLsl",
                        rs.getString("PREV_LSL"));

                row.put("currentLsl",
                        rs.getString("LSL"));

                row.put("previousUsl",
                        rs.getString("PREV_USL"));

                row.put("currentUsl",
                        rs.getString("USL"));

                row.put("changeType",
                        rs.getString("CHANGE_TYPE"));

                result.add(row);
            }

            new ObjectMapper()
                    .writeValue(response.getWriter(), result);

        } catch (Exception e) {

            response.setStatus(500);

            Map<String, String> error =
                    new HashMap<>();

            error.put("error", e.getMessage());

            new ObjectMapper()
                    .writeValue(response.getWriter(), error);
        }
    }
}
