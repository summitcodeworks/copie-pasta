import com.fasterxml.jackson.databind.ObjectMapper;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.sql.*;
import java.util.*;

@WebServlet("/parameter-changes")
public class ParameterChangeServlet extends HttpServlet {

    private static final String URL =
            "jdbc:oracle:thin:@//hostname:1521/service_name";

    private static final String USERNAME = "username";
    private static final String PASSWORD = "password";

    private static final String SQL =
            "WITH data AS ( " +
            "    SELECT " +
            "        product_id, " +
            "        module_id, " +
            "        parameter_name, " +
            "        create_dtts, " +
            "        LAG(create_dtts) OVER ( " +
            "            PARTITION BY product_id, module_id, parameter_name " +
            "            ORDER BY create_dtts " +
            "        ) last_changed_date, " +
            "        value, " +
            "        LAG(value) OVER ( " +
            "            PARTITION BY product_id, module_id, parameter_name " +
            "            ORDER BY create_dtts " +
            "        ) prev_value, " +
            "        lsl, " +
            "        LAG(lsl) OVER ( " +
            "            PARTITION BY product_id, module_id, parameter_name " +
            "            ORDER BY create_dtts " +
            "        ) prev_lsl, " +
            "        usl, " +
            "        LAG(usl) OVER ( " +
            "            PARTITION BY product_id, module_id, parameter_name " +
            "            ORDER BY create_dtts " +
            "        ) prev_usl " +
            "    FROM parameter_history " +
            "    WHERE create_dtts >= SYSDATE - 7 " +
            "), " +
            "changes AS ( " +
            "    SELECT " +
            "        d.*, " +
            "        CASE " +
            "            WHEN NVL(prev_value,'~') <> NVL(value,'~') THEN 'VALUE_CHANGED' " +
            "            WHEN NVL(prev_lsl,-999999999) <> NVL(lsl,-999999999) THEN 'LSL_CHANGED' " +
            "            WHEN NVL(prev_usl,-999999999) <> NVL(usl,-999999999) THEN 'USL_CHANGED' " +
            "        END change_type, " +
            "        ROW_NUMBER() OVER ( " +
            "            PARTITION BY product_id, module_id, parameter_name " +
            "            ORDER BY create_dtts DESC " +
            "        ) rn " +
            "    FROM data d " +
            "    WHERE " +
            "           NVL(prev_value,'~') <> NVL(value,'~') " +
            "        OR NVL(prev_lsl,-999999999) <> NVL(lsl,-999999999) " +
            "        OR NVL(prev_usl,-999999999) <> NVL(usl,-999999999) " +
            ") " +
            "SELECT " +
            "    product_id, " +
            "    module_id, " +
            "    parameter_name, " +
            "    last_changed_date, " +
            "    create_dtts current_changed_date, " +
            "    prev_value, " +
            "    value, " +
            "    prev_lsl, " +
            "    lsl, " +
            "    prev_usl, " +
            "    usl, " +
            "    change_type " +
            "FROM changes " +
            "WHERE rn = 1 " +
            "ORDER BY product_id, module_id, parameter_name";

    @Override
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response)
            throws ServletException, IOException {

        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");

        List<Map<String, Object>> data = new ArrayList<Map<String, Object>>();

        Connection conn = null;
        PreparedStatement ps = null;
        ResultSet rs = null;

        try {

            Class.forName("oracle.jdbc.driver.OracleDriver");

            conn = DriverManager.getConnection(
                    URL,
                    USERNAME,
                    PASSWORD
            );

            ps = conn.prepareStatement(SQL);

            rs = ps.executeQuery();

            while (rs.next()) {

                Map<String, Object> row =
                        new LinkedHashMap<String, Object>();

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

                data.add(row);
            }

            Map<String, Object> result =
                    new HashMap<String, Object>();

            result.put("data", data);

            ObjectMapper mapper = new ObjectMapper();

            mapper.writeValue(
                    response.getWriter(),
                    result
            );

        } catch (Exception e) {

            response.setStatus(
                    HttpServletResponse.SC_INTERNAL_SERVER_ERROR
            );

            Map<String, String> error =
                    new HashMap<String, String>();

            error.put("error", e.getMessage());

            new ObjectMapper().writeValue(
                    response.getWriter(),
                    error
            );

        } finally {

            try {
                if (rs != null) rs.close();
            } catch (Exception ignored) {}

            try {
                if (ps != null) ps.close();
            } catch (Exception ignored) {}

            try {
                if (conn != null) conn.close();
            } catch (Exception ignored) {}
        }
    }
}
