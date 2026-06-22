package com.example.report;

import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import javax.naming.InitialContext;
import javax.naming.NamingException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.sql.DataSource;

@WebServlet(urlPatterns = {
    "/api/parameter-changes",
    "/api/parameter-change-summary",
    "/api/cascading-options"
})
public class ParameterChangesServlet extends HttpServlet {
  private static final long serialVersionUID = 1L;
  private static final String DATASOURCE_NAME = "java:comp/env/jdbc/ParameterDb";
  private static final int DEFAULT_PAGE_SIZE = 50;
  private static final int MAX_PAGE_SIZE = 200;

  @Override
  protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
    response.setContentType("application/json");
    response.setCharacterEncoding("UTF-8");
    Connection connection = null;
    try {
      connection = getConnection();
      if ("/api/cascading-options".equals(request.getServletPath())) {
        writeCascadingOptions(connection, request, response);
      } else if ("/api/parameter-change-summary".equals(request.getServletPath())) {
        writeParameterChangeSummary(connection, request, response);
      } else {
        writeParameterChanges(connection, request, response);
      }
    } catch (Exception e) {
      response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
      response.getWriter().write("{\"error\":\"" + json(e.getMessage()) + "\"}");
    } finally {
      closeConnection(connection);
    }
  }

  private void writeParameterChangeSummary(Connection connection, HttpServletRequest request,
      HttpServletResponse response) throws SQLException, IOException {
    String rcp = querySummary(connection, "parameter_history", "RCP", request);
    String ecm = querySummary(connection, "parameter_history_2", "ECM", request);
    response.getWriter().write("{\"data\":[" + rcp
        + (rcp.length() > 0 && ecm.length() > 0 ? "," : "") + ecm + "]}");
  }

  private String querySummary(Connection connection, String tableName, String source,
      HttpServletRequest request) throws SQLException {
    List<Object> params = new ArrayList<Object>();
    String sql = buildSummarySql(tableName, request, params);
    StringBuilder out = new StringBuilder();
    PreparedStatement ps = null;
    ResultSet rs = null;
    try {
      ps = connection.prepareStatement(sql);
      bindParams(ps, params);
      rs = ps.executeQuery();
      while (rs.next()) {
        if (out.length() > 0) out.append(",");
        out.append("{");
        addJson(out, "source", source, true);
        addJson(out, "moduleId", rs.getString("module_id"), false);
        addJson(out, "parameterName", rs.getString("param_name"), false);
        addJson(out, "firstChangeTime", formatDate(rs.getTimestamp("first_change_time")), false);
        addJson(out, "lastChangeTime", formatDate(rs.getTimestamp("last_change_time")), false);
        addJsonNumber(out, "valueChangeCount", rs.getLong("value_change_count"));
        addJsonNumber(out, "lslChangeCount", rs.getLong("lsl_change_count"));
        addJsonNumber(out, "uslChangeCount", rs.getLong("usl_change_count"));
        addJsonNumber(out, "totalChangePoints", rs.getLong("total_change_points"));
        addJson(out, "changeType", rs.getString("change_type"), false);
        out.append("}");
      }
    } finally {
      closeResultSet(rs);
      closeStatement(ps);
    }
    return out.toString();
  }

  private String buildSummarySql(String tableName, HttpServletRequest request, List<Object> params) {
    String startDate = compactDate(request.getParameter("startDate"));
    String endDate = compactDate(request.getParameter("endDate"));
    if (startDate == null) startDate = compactDate(daysFromToday(-6));
    if (endDate == null) endDate = compactDate(daysFromToday(0));
    params.add(startDate);
    params.add(endDate);

    StringBuilder filters = new StringBuilder();
    addInFilter(filters, params, "ph.site", request.getParameterValues("site"));
    addInFilter(filters, params, "ph.line", request.getParameterValues("line"));
    addInFilter(filters, params, "ph.module_id", request.getParameterValues("moduleId"));
    addInFilter(filters, params, "ph.product_id", request.getParameterValues("productId"));
    addLikeFilter(filters, params, "ph.parameter_name", request.getParameter("parameterName"));

    return "WITH p AS (SELECT TO_DATE(?, 'DDMMYYYY') start_dt, TO_DATE(?, 'DDMMYYYY') + 1 end_dt FROM dual), d AS (SELECT ph.ROWID rid, ph.module_id, ph.parameter_name param_name, ph.create_dtts, ph.value, LEAD(ph.value) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_value, ph.lsl, LEAD(ph.lsl) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_lsl, ph.usl, LEAD(ph.usl) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_usl, LEAD(ph.create_dtts) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_create_dtts FROM " + tableName + " ph CROSS JOIN p WHERE ph.create_dtts >= p.start_dt AND ph.create_dtts < p.end_dt" + filters + "), c AS (SELECT module_id, param_name, create_dtts, CASE WHEN DECODE(prev_value, value, 0, 1) = 1 THEN 1 ELSE 0 END value_changed, CASE WHEN DECODE(prev_lsl, lsl, 0, 1) = 1 THEN 1 ELSE 0 END lsl_changed, CASE WHEN DECODE(prev_usl, usl, 0, 1) = 1 THEN 1 ELSE 0 END usl_changed FROM d WHERE prev_create_dtts IS NOT NULL AND prev_value IS NOT NULL AND prev_lsl IS NOT NULL AND prev_usl IS NOT NULL AND value IS NOT NULL AND lsl IS NOT NULL AND usl IS NOT NULL AND (DECODE(prev_value, value, 0, 1) = 1 OR DECODE(prev_lsl, lsl, 0, 1) = 1 OR DECODE(prev_usl, usl, 0, 1) = 1)) SELECT module_id, param_name, MIN(create_dtts) first_change_time, MAX(create_dtts) last_change_time, SUM(value_changed) value_change_count, SUM(lsl_changed) lsl_change_count, SUM(usl_changed) usl_change_count, COUNT(*) total_change_points, RTRIM(CASE WHEN SUM(value_changed) > 0 THEN 'VALUE_CHANGED|' END || CASE WHEN SUM(lsl_changed) > 0 THEN 'LSL_CHANGED|' END || CASE WHEN SUM(usl_changed) > 0 THEN 'USL_CHANGED|' END, '|') change_type FROM c GROUP BY module_id, param_name ORDER BY module_id, param_name";
  }

  private Connection getConnection() throws NamingException, SQLException {
    return ((DataSource) new InitialContext().lookup(DATASOURCE_NAME)).getConnection();
  }

  private void writeParameterChanges(Connection connection, HttpServletRequest request,
      HttpServletResponse response) throws SQLException, IOException {
    int page = positiveInt(request.getParameter("page"), 1);
    int pageSize = Math.min(positiveInt(request.getParameter("pageSize"), DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE);
    PagedRows rcp = queryPage(connection, "parameter_history", "RCP", request, page, pageSize);
    PagedRows ecm = queryPage(connection, "parameter_history_2", "ECM", request, page, pageSize);

    StringBuilder out = new StringBuilder("{\"data\":[");
    out.append(rcp.json);
    if (rcp.count > 0 && ecm.count > 0) out.append(",");
    out.append(ecm.json);
    out.append("],\"page\":").append(page)
        .append(",\"pageSize\":").append(pageSize)
        .append(",\"hasMore\":").append(rcp.hasMore || ecm.hasMore)
        .append("}");
    response.getWriter().write(out.toString());
  }

  private PagedRows queryPage(Connection connection, String tableName, String source,
      HttpServletRequest request, int page, int pageSize) throws SQLException {
    List<Object> params = new ArrayList<Object>();
    String sql = buildChangesSql(tableName, request, params, page, pageSize);
    StringBuilder out = new StringBuilder();
    int count = 0;
    PreparedStatement ps = null;
    ResultSet rs = null;
    try {
      ps = connection.prepareStatement(sql);
      bindParams(ps, params);
      rs = ps.executeQuery();
      while (rs.next()) {
        if (count > 0) out.append(",");
        count++;
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
    } finally {
      closeResultSet(rs);
      closeStatement(ps);
    }
    return new PagedRows(out.toString(), count, count == pageSize);
  }

  private String buildChangesSql(String tableName, HttpServletRequest request, List<Object> params,
      int page, int pageSize) {
    String startDate = compactDate(request.getParameter("startDate"));
    String endDate = compactDate(request.getParameter("endDate"));
    if (startDate == null) startDate = compactDate(daysFromToday(-6));
    if (endDate == null) endDate = compactDate(daysFromToday(0));
    params.add(startDate);
    params.add(endDate);

    StringBuilder filters = new StringBuilder();
    addInFilter(filters, params, "ph.site", request.getParameterValues("site"));
    addInFilter(filters, params, "ph.line", request.getParameterValues("line"));
    addInFilter(filters, params, "ph.module_id", request.getParameterValues("moduleId"));
    addInFilter(filters, params, "ph.product_id", request.getParameterValues("productId"));
    addLikeFilter(filters, params, "ph.parameter_name", request.getParameter("parameterName"));
    params.add(Integer.valueOf((page - 1) * pageSize));
    params.add(Integer.valueOf(pageSize));

    return "WITH p AS (SELECT TO_DATE(?, 'DDMMYYYY') start_dt, TO_DATE(?, 'DDMMYYYY') + 1 end_dt FROM dual), d AS (SELECT ph.ROWID rid, ph.site, ph.line, ph.product_id, ph.module_id, ph.parameter_name param_name, ph.create_dtts, ph.value, LEAD(ph.value) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_value, ph.lsl, LEAD(ph.lsl) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_lsl, ph.usl, LEAD(ph.usl) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_usl, LEAD(ph.create_dtts) OVER (PARTITION BY ph.module_id, ph.parameter_name ORDER BY ph.create_dtts DESC, ph.ROWID DESC) prev_create_dtts FROM " + tableName + " ph CROSS JOIN p WHERE ph.create_dtts >= p.start_dt AND ph.create_dtts < p.end_dt" + filters + "), c AS (SELECT d.*, CASE WHEN DECODE(prev_value, value, 0, 1) = 1 THEN 1 ELSE 0 END value_changed, CASE WHEN DECODE(prev_lsl, lsl, 0, 1) = 1 THEN 1 ELSE 0 END lsl_changed, CASE WHEN DECODE(prev_usl, usl, 0, 1) = 1 THEN 1 ELSE 0 END usl_changed FROM d WHERE prev_create_dtts IS NOT NULL AND prev_value IS NOT NULL AND prev_lsl IS NOT NULL AND prev_usl IS NOT NULL AND value IS NOT NULL AND lsl IS NOT NULL AND usl IS NOT NULL AND (DECODE(prev_value, value, 0, 1) = 1 OR DECODE(prev_lsl, lsl, 0, 1) = 1 OR DECODE(prev_usl, usl, 0, 1) = 1)), latest AS (SELECT site, line, product_id, module_id, param_name, MAX(prev_create_dtts) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) last_changed_date, MAX(create_dtts) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) current_changed_date, MAX(CASE WHEN value_changed = 1 THEN prev_value END) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) prev_value, MAX(CASE WHEN value_changed = 1 THEN value END) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) value, MAX(CASE WHEN lsl_changed = 1 THEN prev_lsl END) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) prev_lsl, MAX(CASE WHEN lsl_changed = 1 THEN lsl END) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) lsl, MAX(CASE WHEN usl_changed = 1 THEN prev_usl END) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) prev_usl, MAX(CASE WHEN usl_changed = 1 THEN usl END) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) usl, MAX(value_changed) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) value_changed, MAX(lsl_changed) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) lsl_changed, MAX(usl_changed) KEEP (DENSE_RANK LAST ORDER BY create_dtts, rid) usl_changed FROM c GROUP BY site, line, product_id, module_id, param_name) SELECT site, line, product_id, module_id, param_name parameter_name, last_changed_date, current_changed_date, prev_value, value, prev_lsl, lsl, prev_usl, usl, RTRIM(CASE WHEN value_changed = 1 THEN 'VALUE_CHANGED|' END || CASE WHEN lsl_changed = 1 THEN 'LSL_CHANGED|' END || CASE WHEN usl_changed = 1 THEN 'USL_CHANGED|' END, '|') change_type FROM latest ORDER BY module_id, param_name, product_id OFFSET ? ROWS FETCH NEXT ? ROWS ONLY";
  }

  private void writeCascadingOptions(Connection connection, HttpServletRequest request,
      HttpServletResponse response) throws SQLException, IOException {
    response.getWriter().write("{\"sites\":" + queryOptions(connection, "site", request)
        + ",\"lines\":" + queryOptions(connection, "line", request)
        + ",\"modules\":" + queryOptions(connection, "module_id", request) + "}");
  }

  private String queryOptions(Connection connection, String field, HttpServletRequest request)
      throws SQLException {
    List<Object> params = new ArrayList<Object>();
    StringBuilder filters = new StringBuilder();
    if (!"site".equals(field)) addInFilter(filters, params, "site", request.getParameterValues("site"));
    if (!"line".equals(field)) addInFilter(filters, params, "line", request.getParameterValues("line"));
    if (!"module_id".equals(field)) addInFilter(filters, params, "module_id", request.getParameterValues("moduleId"));
    String sql = "SELECT DISTINCT " + field + " value FROM parameter_history WHERE " + field + " IS NOT NULL" + filters + " UNION SELECT DISTINCT " + field + " value FROM parameter_history_2 WHERE " + field + " IS NOT NULL" + filters + " ORDER BY value";
    List<Object> allParams = new ArrayList<Object>(params);
    allParams.addAll(params);
    StringBuilder out = new StringBuilder("[");
    boolean first = true;
    PreparedStatement ps = null;
    ResultSet rs = null;
    try {
      ps = connection.prepareStatement(sql);
      bindParams(ps, allParams);
      rs = ps.executeQuery();
      while (rs.next()) {
        if (!first) out.append(",");
        first = false;
        out.append("\"").append(json(rs.getString("value"))).append("\"");
      }
    } finally {
      closeResultSet(rs);
      closeStatement(ps);
    }
    return out.append("]").toString();
  }

  private void addInFilter(StringBuilder sql, List<Object> params, String column, String[] values) {
    List<String> clean = splitRequestValues(values);
    if (clean.isEmpty()) return;
    String joined = joinValues(clean);
    sql.append(" AND ").append(column)
        .append(" IN (SELECT TRIM(REGEXP_SUBSTR(?,'[^,]+',1,LEVEL)) FROM dual")
        .append(" CONNECT BY REGEXP_SUBSTR(?,'[^,]+',1,LEVEL) IS NOT NULL)");
    params.add(joined);
    params.add(joined);
  }

  private void addLikeFilter(StringBuilder sql, List<Object> params, String column, String value) {
    if (value != null && value.trim().length() > 0) {
      sql.append(" AND LOWER(").append(column).append(") LIKE ?");
      params.add("%" + value.trim().toLowerCase() + "%");
    }
  }

  private List<String> splitRequestValues(String[] requestValues) {
    List<String> values = new ArrayList<String>();
    if (requestValues == null) return values;
    for (int i = 0; i < requestValues.length; i++) {
      String requestValue = requestValues[i];
      if (requestValue == null) continue;
      String[] parts = requestValue.split(",");
      for (int j = 0; j < parts.length; j++) {
        String part = parts[j];
        String item = part.trim();
        if (item.length() > 0 && !"All".equalsIgnoreCase(item)) values.add(item);
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

  private void bindParams(PreparedStatement ps, List<Object> params) throws SQLException {
    for (int i = 0; i < params.size(); i++) ps.setObject(i + 1, params.get(i));
  }

  private void closeResultSet(ResultSet rs) {
    if (rs != null) {
      try {
        rs.close();
      } catch (SQLException ignored) {
      }
    }
  }

  private void closeStatement(PreparedStatement ps) {
    if (ps != null) {
      try {
        ps.close();
      } catch (SQLException ignored) {
      }
    }
  }

  private void closeConnection(Connection connection) {
    if (connection != null) {
      try {
        connection.close();
      } catch (SQLException ignored) {
      }
    }
  }

  private int positiveInt(String value, int fallback) {
    try {
      int parsed = Integer.parseInt(value);
      return parsed > 0 ? parsed : fallback;
    } catch (Exception e) {
      return fallback;
    }
  }

  private String compactDate(String value) {
    if (value == null) return null;
    if (value.matches("\\d{4}-\\d{2}-\\d{2}")) {
      return value.substring(8, 10) + value.substring(5, 7) + value.substring(0, 4);
    }
    String digits = value.replaceAll("[^0-9]", "");
    return digits.length() == 8 ? digits : null;
  }

  private String daysFromToday(int days) {
    Calendar calendar = Calendar.getInstance();
    calendar.add(Calendar.DAY_OF_MONTH, days);
    return new SimpleDateFormat("dd/MM/yyyy").format(calendar.getTime());
  }

  private String formatDate(java.sql.Timestamp value) {
    return value == null ? "" : new SimpleDateFormat("dd/MM/yyyy HH:mm:ss").format(value);
  }

  private void addJson(StringBuilder out, String key, String value, boolean first) {
    if (!first) out.append(",");
    out.append("\"").append(key).append("\":\"").append(json(value)).append("\"");
  }

  private void addJsonNumber(StringBuilder out, String key, long value) {
    out.append(",\"").append(key).append("\":").append(value);
  }

  private String json(String value) {
    if (value == null) return "";
    return value.replace("\\", "\\\\").replace("\"", "\\\"")
        .replace("\r", "\\r").replace("\n", "\\n");
  }

  private static class PagedRows {
    final String json;
    final int count;
    final boolean hasMore;
    PagedRows(String json, int count, boolean hasMore) {
      this.json = json;
      this.count = count;
      this.hasMore = hasMore;
    }
  }
}
