package com.example.report;

import java.io.IOException;
import java.sql.Connection;
import java.sql.Date;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import javax.naming.InitialContext;
import javax.naming.NamingException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.sql.DataSource;

@WebServlet(urlPatterns = {"/api/parameter-changes", "/api/cascading-options"})
public class ParameterChangesServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;
  private static final String DATASOURCE_NAME = "java:comp/env/jdbc/ParameterDb";

  @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
    response.setContentType("application/json");
    response.setCharacterEncoding("UTF-8");

    try (Connection connection = getConnection()) {
      String path = request.getServletPath();
      if ("/api/cascading-options".equals(path)) {
        writeCascadingOptions(connection, request, response);
      } else {
        writeParameterChanges(connection, request, response);
      }
    } catch (Exception e) {
      response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
      response.getWriter().write("{\"error\":\"" + json(e.getMessage()) + "\"}");
    }
  }

  private Connection getConnection() throws NamingException, SQLException {
    DataSource dataSource = (DataSource) new InitialContext().lookup(DATASOURCE_NAME);
    return dataSource.getConnection();
  }

  private void writeParameterChanges(Connection connection, HttpServletRequest request, HttpServletResponse response)
      throws SQLException, IOException, ParseException {
    List<Object> params1 = new ArrayList<Object>();
    List<Object> params2 = new ArrayList<Object>();

    String sql1 = buildChangesSql("parameter_history", request, params1);
    String sql2 = buildChangesSql("parameter_history_2", request, params2);

    StringBuilder out = new StringBuilder();
    out.append("{\"data1\":");
    out.append(queryRows(connection, sql1, params1, "Table 1"));
    out.append(",\"data2\":");
    out.append(queryRows(connection, sql2, params2, "Table 2"));
    out.append("}");
    response.getWriter().write(out.toString());
  }

  private String buildChangesSql(String tableName, HttpServletRequest request, List<Object> params)
      throws ParseException {
    StringBuilder filters = new StringBuilder();
    addInFilter(filters, params, "site", request.getParameterValues("site"));
    addInFilter(filters, params, "line", request.getParameterValues("line"));
    addInFilter(filters, params, "module_id", request.getParameterValues("moduleId"));
    addInFilter(filters, params, "product_id", request.getParameterValues("productId"));
    addLikeFilter(filters, params, "parameter_name", request.getParameter("parameterName"));
    addDateFilter(filters, params, request);

    return ""
        + "WITH data AS ("
        + " SELECT site, line, product_id, module_id, parameter_name, create_dtts,"
        + " LAG(create_dtts) OVER (PARTITION BY product_id, module_id, parameter_name ORDER BY create_dtts) last_changed_date,"
        + " value,"
        + " LAG(value) OVER (PARTITION BY product_id, module_id, parameter_name ORDER BY create_dtts) prev_value,"
        + " lsl,"
        + " LAG(lsl) OVER (PARTITION BY product_id, module_id, parameter_name ORDER BY create_dtts) prev_lsl,"
        + " usl,"
        + " LAG(usl) OVER (PARTITION BY product_id, module_id, parameter_name ORDER BY create_dtts) prev_usl"
        + " FROM " + tableName
        + " WHERE 1 = 1"
        + filters
        + "), changes AS ("
        + " SELECT d.*, RTRIM("
        + " CASE WHEN NVL(prev_value, '~') <> NVL(value, '~') THEN 'VALUE_CHANGED|' END"
        + " || CASE WHEN NVL(prev_lsl, -999999999) <> NVL(lsl, -999999999) THEN 'LSL_CHANGED|' END"
        + " || CASE WHEN NVL(prev_usl, -999999999) <> NVL(usl, -999999999) THEN 'USL_CHANGED|' END,"
        + " '|') change_type,"
        + " ROW_NUMBER() OVER (PARTITION BY product_id, module_id, parameter_name ORDER BY create_dtts DESC) rn"
        + " FROM data d"
        + " WHERE NVL(prev_value, '~') <> NVL(value, '~')"
        + " OR NVL(prev_lsl, -999999999) <> NVL(lsl, -999999999)"
        + " OR NVL(prev_usl, -999999999) <> NVL(usl, -999999999)"
        + ")"
        + " SELECT site, line, product_id, module_id, parameter_name,"
        + " last_changed_date, create_dtts current_changed_date,"
        + " prev_value, value, prev_lsl, lsl, prev_usl, usl, change_type"
        + " FROM changes"
        + " WHERE rn = 1"
        + " ORDER BY product_id, module_id, parameter_name";
  }

  private String queryRows(Connection connection, String sql, List<Object> params, String source)
      throws SQLException {
    StringBuilder out = new StringBuilder("[");
    boolean first = true;

    try (PreparedStatement ps = connection.prepareStatement(sql)) {
      bindParams(ps, params);
      try (ResultSet rs = ps.executeQuery()) {
        while (rs.next()) {
          if (!first) out.append(",");
          first = false;
          out.append("{");
          addJson(out, "source", source, true);
          addJson(out, "site", rs.getString("site"), false);
          addJson(out, "line", rs.getString("line"), false);
          addJson(out, "productId", rs.getString("product_id"), false);
          addJson(out, "moduleId", rs.getString("module_id"), false);
          addJson(out, "parameterName", rs.getString("parameter_name"), false);
          addJson(out, "createDttd", formatDate(rs.getTimestamp("current_changed_date")), false);
          addJson(out, "previousValue", rs.getString("prev_value"), false);
          addJson(out, "currentValue", rs.getString("value"), false);
          addJson(out, "previousLsl", rs.getString("prev_lsl"), false);
          addJson(out, "currentLsl", rs.getString("lsl"), false);
          addJson(out, "previousUsl", rs.getString("prev_usl"), false);
          addJson(out, "currentUsl", rs.getString("usl"), false);
          addJson(out, "changeType", rs.getString("change_type"), false);
          out.append("}");
        }
      }
    }

    out.append("]");
    return out.toString();
  }

  private void writeCascadingOptions(Connection connection, HttpServletRequest request, HttpServletResponse response)
      throws SQLException {
    response.getWriter().write("{"
        + "\"sites\":" + queryOptions(connection, "site", request)
        + ",\"lines\":" + queryOptions(connection, "line", request)
        + ",\"modules\":" + queryOptions(connection, "module_id", request)
        + "}");
  }

  private String queryOptions(Connection connection, String field, HttpServletRequest request) throws SQLException {
    List<Object> params = new ArrayList<Object>();
    StringBuilder filters = new StringBuilder();
    if (!"site".equals(field)) addInFilter(filters, params, "site", request.getParameterValues("site"));
    if (!"line".equals(field)) addInFilter(filters, params, "line", request.getParameterValues("line"));
    if (!"module_id".equals(field)) addInFilter(filters, params, "module_id", request.getParameterValues("moduleId"));

    String sql = "SELECT DISTINCT " + field + " value FROM parameter_history WHERE " + field
        + " IS NOT NULL" + filters
        + " UNION SELECT DISTINCT " + field + " value FROM parameter_history_2 WHERE " + field
        + " IS NOT NULL" + filters
        + " ORDER BY value";

    List<Object> allParams = new ArrayList<Object>();
    allParams.addAll(params);
    allParams.addAll(params);

    StringBuilder out = new StringBuilder("[");
    boolean first = true;
    try (PreparedStatement ps = connection.prepareStatement(sql)) {
      bindParams(ps, allParams);
      try (ResultSet rs = ps.executeQuery()) {
        while (rs.next()) {
          if (!first) out.append(",");
          first = false;
          out.append("\"").append(json(rs.getString("value"))).append("\"");
        }
      }
    }
    out.append("]");
    return out.toString();
  }

  private void addInFilter(StringBuilder sql, List<Object> params, String column, String[] requestValues) {
    List<String> values = splitRequestValues(requestValues);
    if (values.isEmpty()) {
      return;
    }

    String valueList = joinValues(values);
    sql.append(" AND ").append(column)
        .append(" IN (")
        .append(" SELECT TRIM(REGEXP_SUBSTR(?, '[^,]+', 1, LEVEL))")
        .append(" FROM dual")
        .append(" CONNECT BY REGEXP_SUBSTR(?, '[^,]+', 1, LEVEL) IS NOT NULL")
        .append(")");
    params.add(valueList);
    params.add(valueList);
  }

  private List<String> splitRequestValues(String[] requestValues) {
    List<String> values = new ArrayList<String>();
    if (requestValues == null) {
      return values;
    }

    for (int i = 0; i < requestValues.length; i++) {
      if (requestValues[i] == null) continue;
      String[] parts = requestValues[i].split(",");
      for (int j = 0; j < parts.length; j++) {
        String item = parts[j].trim();
        if (item.length() > 0 && !"All".equalsIgnoreCase(item)) {
          values.add(item);
        }
      }
    }
    return values;
  }

  private String joinValues(List<String> values) {
    StringBuilder out = new StringBuilder();
    for (int i = 0; i < values.size(); i++) {
      if (i > 0) out.append(",");
      out.append(values.get(i));
    }
    return out.toString();
  }

  private void addLikeFilter(StringBuilder sql, List<Object> params, String column, String value) {
    if (value != null && value.trim().length() > 0) {
      sql.append(" AND LOWER(").append(column).append(") LIKE ?");
      params.add("%" + value.trim().toLowerCase() + "%");
    }
  }

  private void addDateFilter(StringBuilder sql, List<Object> params, HttpServletRequest request)
      throws ParseException {
    SimpleDateFormat sdf = new SimpleDateFormat("dd/MM/yyyy");
    String startDate = firstValue(request.getParameter("startDate"), request.getParameter("fromDate"));
    String endDate = firstValue(request.getParameter("endDate"), request.getParameter("toDate"));
    if (startDate != null && startDate.trim().length() > 0) {
      sql.append(" AND create_dtts >= ?");
      params.add(new Date(sdf.parse(startDate.trim()).getTime()));
    }
    if (endDate != null && endDate.trim().length() > 0) {
      sql.append(" AND create_dtts < ? + INTERVAL '1' DAY");
      params.add(new Date(sdf.parse(endDate.trim()).getTime()));
    }
  }

  private String firstValue(String preferred, String fallback) {
    if (preferred != null && preferred.trim().length() > 0) {
      return preferred;
    }
    return fallback;
  }

  private void bindParams(PreparedStatement ps, List<Object> params) throws SQLException {
    for (int i = 0; i < params.size(); i++) {
      ps.setObject(i + 1, params.get(i));
    }
  }

  private String formatDate(java.sql.Timestamp value) {
    if (value == null) return "";
    return new SimpleDateFormat("dd/MM/yyyy").format(value);
  }

  private void addJson(StringBuilder out, String key, String value, boolean first) {
    if (!first) out.append(",");
    out.append("\"").append(key).append("\":\"").append(json(value)).append("\"");
  }

  private String json(String value) {
    if (value == null) return "";
    return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", "\\r").replace("\n", "\\n");
  }
}
