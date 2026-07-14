package com.example.report;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;
import javax.servlet.ServletContext;
import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;
import javax.servlet.annotation.WebListener;

@WebListener
public class ParameterChangesScheduler implements ServletContextListener {
  private static final long HOUR_MILLIS = 60L * 60L * 1000L;

  private static final String DEFAULT_DRIVER = "oracle.jdbc.OracleDriver";
  private static final String DEFAULT_MAIN_TABLE = "parameter_change_results";
  private static final String DEFAULT_SUMMARY_TABLE = "parameter_change_summary_results";
  private static final String DEFAULT_TREND_TABLE = "parameter_change_trend_results";

  private Timer timer;

  public void contextInitialized(ServletContextEvent event) {
    ServletContext context = event.getServletContext();
    long intervalMillis = getLong(context, "parameter.scheduler.intervalMillis", HOUR_MILLIS);
    timer = new Timer("parameter-changes-scheduler", true);
    timer.scheduleAtFixedRate(new ParameterChangesTask(context), 0L, intervalMillis);
  }

  public void contextDestroyed(ServletContextEvent event) {
    if (timer != null) {
      timer.cancel();
      timer = null;
    }
  }

  private static final class ParameterChangesTask extends TimerTask {
    private final ServletContext context;

    ParameterChangesTask(ServletContext context) {
      this.context = context;
    }

    public void run() {
      try {
        String driver = getConfig(context, "parameter.jdbc.driver", "PARAMETER_JDBC_DRIVER", DEFAULT_DRIVER);
        Class.forName(driver);

        try (Connection source = openConnection(context,
            "parameter.source.jdbc.url", "PARAMETER_SOURCE_JDBC_URL",
            "parameter.source.jdbc.user", "PARAMETER_SOURCE_JDBC_USER",
            "parameter.source.jdbc.password", "PARAMETER_SOURCE_JDBC_PASSWORD");
            Connection target = openConnection(context,
            "parameter.target.jdbc.url", "PARAMETER_TARGET_JDBC_URL",
            "parameter.target.jdbc.user", "PARAMETER_TARGET_JDBC_USER",
            "parameter.target.jdbc.password", "PARAMETER_TARGET_JDBC_PASSWORD")) {
          target.setAutoCommit(false);

          try {
            String mainTableRCP = getConfig(context, "parameter.target.mainTableRCP",
                "PARAMETER_TARGET_MAIN_TABLE_RCP", DEFAULT_MAIN_TABLE + "_rcp");
            String summaryTableRCP = getConfig(context, "parameter.target.summaryTableRCP",
                "PARAMETER_TARGET_SUMMARY_TABLE_RCP", DEFAULT_SUMMARY_TABLE + "_rcp");
            String trendTableRCP = getConfig(context, "parameter.target.trendTableRCP",
                "PARAMETER_TARGET_TREND_TABLE_RCP", DEFAULT_TREND_TABLE + "_rcp");

            String mainTableECM = getConfig(context, "parameter.target.mainTableECM",
                "PARAMETER_TARGET_MAIN_TABLE_ECM", DEFAULT_MAIN_TABLE + "_ecm");
            String summaryTableECM = getConfig(context, "parameter.target.summaryTableECM",
                "PARAMETER_TARGET_SUMMARY_TABLE_ECM", DEFAULT_SUMMARY_TABLE + "_ecm");
            String trendTableECM = getConfig(context, "parameter.target.trendTableECM",
                "PARAMETER_TARGET_TREND_TABLE_ECM", DEFAULT_TREND_TABLE + "_ecm");

            DateRange range = lastHourRange();
            Timestamp runTime = new Timestamp(System.currentTimeMillis());
            CopyResult total = new CopyResult();
            total.add(copyAllResults(source, target, mainTableRCP, summaryTableRCP, trendTableRCP,
                "PARAMETER_HISTORY", "RCP", runTime, range));
            total.add(copyAllResults(source, target, mainTableECM, summaryTableECM, trendTableECM,
                "PARAMETER_HISTORY_2", "ECM", runTime, range));
            target.commit();
            context.log("Parameter changes scheduler inserted " + total);
          } catch (Exception e) {
            rollback(target);
            throw e;
          }
        }
      } catch (Exception e) {
        context.log("Parameter changes scheduler failed", e);
      }
    }

    private CopyResult copyAllResults(Connection source, Connection target, String mainTable,
        String summaryTable, String trendTable, String sourceTable, String sourceName, Timestamp runTime,
        DateRange range) throws SQLException {
      CopyResult result = copyMainChanges(source, target, mainTable, sourceTable, sourceName, runTime, range);
      for (ModuleParameter key : result.keys) {
        result.summaryRows += copySummary(source, target, summaryTable, sourceTable, sourceName, runTime,
            range, key.moduleId, key.paramName);
        result.trendRows += copyTrend(source, target, trendTable, sourceTable, sourceName, runTime,
            range, key.moduleId, key.paramName);
      }
      return result;
    }

    private CopyResult copyMainChanges(Connection source, Connection target, String targetTable,
        String sourceTable, String sourceName, Timestamp runTime, DateRange range) throws SQLException {
      CopyResult result = new CopyResult();
      try (PreparedStatement read = source.prepareStatement(buildMainChangesSql(sourceTable));
          PreparedStatement write = target.prepareStatement(buildMainInsertSql(targetTable))) {
        bindDateRange(read, range);
        try (ResultSet rs = read.executeQuery()) {
          while (rs.next()) {
            String moduleId = rs.getString("MODULE_ID");
            String paramName = rs.getString("PARAM_NAME");
            int i = 1;
            write.setTimestamp(i++, runTime);
            write.setString(i++, sourceName);
            write.setString(i++, moduleId);
            write.setString(i++, paramName);
            write.setTimestamp(i++, rs.getTimestamp("PREVIOUS_CHANGE_TIME"));
            write.setTimestamp(i++, rs.getTimestamp("CURRENT_CHANGE_TIME"));
            write.setString(i++, rs.getString("PREV_VALUE"));
            write.setString(i++, rs.getString("VALUE"));
            write.setString(i++, rs.getString("PREV_LSL"));
            write.setString(i++, rs.getString("LSL"));
            write.setString(i++, rs.getString("PREV_USL"));
            write.setString(i++, rs.getString("USL"));
            write.setString(i++, rs.getString("CHANGE_TYPE"));
            write.addBatch();
            result.mainRows++;
            result.addKey(moduleId, paramName);
          }
        }
        write.executeBatch();
      }
      return result;
    }

    private int copySummary(Connection source, Connection target, String targetTable, String sourceTable,
        String sourceName, Timestamp runTime, DateRange range, String moduleId, String paramName)
        throws SQLException {
      int count = 0;
      try (PreparedStatement read = source.prepareStatement(buildSummarySql(sourceTable));
          PreparedStatement write = target.prepareStatement(buildSummaryInsertSql(targetTable))) {
        bindDateRangeModuleParam(read, range, moduleId, paramName);
        try (ResultSet rs = read.executeQuery()) {
          while (rs.next()) {
            int i = 1;
            write.setTimestamp(i++, runTime);
            write.setString(i++, sourceName);
            write.setString(i++, rs.getString("MODULE_ID"));
            write.setString(i++, rs.getString("PARAM_NAME"));
            write.setTimestamp(i++, rs.getTimestamp("FIRST_CHANGE_TIME"));
            write.setTimestamp(i++, rs.getTimestamp("LAST_CHANGE_TIME"));
            write.setLong(i++, rs.getLong("VALUE_CHANGE_COUNT"));
            write.setLong(i++, rs.getLong("LSL_CHANGE_COUNT"));
            write.setLong(i++, rs.getLong("USL_CHANGE_COUNT"));
            write.setLong(i++, rs.getLong("TOTAL_CHANGE_POINTS"));
            write.setString(i++, rs.getString("CHANGE_TYPE"));
            write.addBatch();
            count++;
          }
        }
        write.executeBatch();
      }
      return count;
    }

    private int copyTrend(Connection source, Connection target, String targetTable, String sourceTable,
        String sourceName, Timestamp runTime, DateRange range, String moduleId, String paramName)
        throws SQLException {
      int count = 0;
      try (PreparedStatement read = source.prepareStatement(buildTrendSql(sourceTable));
          PreparedStatement write = target.prepareStatement(buildTrendInsertSql(targetTable))) {
        bindDateRangeModuleParam(read, range, moduleId, paramName);
        try (ResultSet rs = read.executeQuery()) {
          while (rs.next()) {
            int i = 1;
            write.setTimestamp(i++, runTime);
            write.setString(i++, sourceName);
            write.setString(i++, rs.getString("MODULE_ID"));
            write.setString(i++, rs.getString("PARAM_NAME"));
            write.setTimestamp(i++, rs.getTimestamp("PREVIOUS_CHANGE_TIME"));
            write.setTimestamp(i++, rs.getTimestamp("CURRENT_CHANGE_TIME"));
            write.setString(i++, rs.getString("PREV_VALUE"));
            write.setString(i++, rs.getString("VALUE"));
            write.setString(i++, rs.getString("PREV_LSL"));
            write.setString(i++, rs.getString("LSL"));
            write.setString(i++, rs.getString("PREV_USL"));
            write.setString(i++, rs.getString("USL"));
            write.setString(i++, rs.getString("CHANGE_TYPE"));
            write.addBatch();
            count++;
          }
        }
        write.executeBatch();
      }
      return count;
    }

    private String buildMainChangesSql(String tableName) {
      return ""
          + "WITH p AS (\n"
          + "    SELECT\n"
          + "        TO_DATE(?, 'DDMMYYYY') AS start_dt,\n"
          + "        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt\n"
          + "    FROM dual\n"
          + "),\n"
          + "d AS (\n"
          + "    SELECT\n"
          + "        ph.ROWID AS rid,\n"
          + "        ph.MODULE_ID,\n"
          + "        ph.PARAM_NAME,\n"
          + "        ph.CREATE_DTTS,\n"
          + "\n"
          + "        ph.VALUE,\n"
          + "        LEAD(ph.VALUE) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_VALUE,\n"
          + "\n"
          + "        ph.LSL,\n"
          + "        LEAD(ph.LSL) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_LSL,\n"
          + "\n"
          + "        ph.USL,\n"
          + "        LEAD(ph.USL) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_USL,\n"
          + "\n"
          + "        LEAD(ph.CREATE_DTTS) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_CREATE_DTTS\n"
          + "\n"
          + "    FROM " + tableName + " ph\n"
          + "    CROSS JOIN p\n"
          + "    WHERE ph.CREATE_DTTS >= p.start_dt\n"
          + "      AND ph.CREATE_DTTS <  p.end_dt\n"
          + "),\n"
          + "c AS (\n"
          + "    SELECT\n"
          + "        d.*,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS VALUE_CHANGED,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS LSL_CHANGED,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_USL, d.USL, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS USL_CHANGED\n"
          + "\n"
          + "    FROM d\n"
          + "    WHERE d.PREV_CREATE_DTTS IS NOT NULL\n"
          + "      AND d.PREV_VALUE IS NOT NULL\n"
          + "      AND d.PREV_LSL IS NOT NULL\n"
          + "      AND d.PREV_USL IS NOT NULL\n"
          + "      AND d.VALUE IS NOT NULL\n"
          + "      AND d.LSL IS NOT NULL\n"
          + "      AND d.USL IS NOT NULL\n"
          + "      AND (\n"
          + "             DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1\n"
          + "          OR DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1\n"
          + "          OR DECODE(d.PREV_USL, d.USL, 0, 1) = 1\n"
          + "      )\n"
          + ")\n"
          + "SELECT\n"
          + "    MODULE_ID,\n"
          + "    PARAM_NAME,\n"
          + "\n"
          + "    PREV_CREATE_DTTS AS PREVIOUS_CHANGE_TIME,\n"
          + "    CREATE_DTTS AS CURRENT_CHANGE_TIME,\n"
          + "\n"
          + "    CASE WHEN VALUE_CHANGED = 1 THEN PREV_VALUE END AS PREV_VALUE,\n"
          + "    CASE WHEN VALUE_CHANGED = 1 THEN VALUE END AS VALUE,\n"
          + "\n"
          + "    CASE WHEN LSL_CHANGED = 1 THEN PREV_LSL END AS PREV_LSL,\n"
          + "    CASE WHEN LSL_CHANGED = 1 THEN LSL END AS LSL,\n"
          + "\n"
          + "    CASE WHEN USL_CHANGED = 1 THEN PREV_USL END AS PREV_USL,\n"
          + "    CASE WHEN USL_CHANGED = 1 THEN USL END AS USL,\n"
          + "\n"
          + "    RTRIM(\n"
          + "          CASE WHEN VALUE_CHANGED = 1 THEN 'VALUE_CHANGED|' END\n"
          + "       || CASE WHEN LSL_CHANGED = 1 THEN 'LSL_CHANGED|' END\n"
          + "       || CASE WHEN USL_CHANGED = 1 THEN 'USL_CHANGED|' END,\n"
          + "       '|'\n"
          + "    ) AS CHANGE_TYPE\n"
          + "\n"
          + "FROM c\n"
          + "ORDER BY\n"
          + "    MODULE_ID,\n"
          + "    PARAM_NAME,\n"
          + "    CURRENT_CHANGE_TIME";
    }

    private String buildTrendSql(String tableName) {
      return ""
          + "WITH p AS (\n"
          + "    SELECT\n"
          + "        TO_DATE(?, 'DDMMYYYY') AS start_dt,\n"
          + "        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt,\n"
          + "        TRIM(?) AS module_id,\n"
          + "        TRIM(?) AS param_name\n"
          + "    FROM dual\n"
          + "),\n"
          + "d AS (\n"
          + "    SELECT\n"
          + "        ph.ROWID AS rid,\n"
          + "        ph.MODULE_ID,\n"
          + "        ph.PARAM_NAME,\n"
          + "        ph.CREATE_DTTS,\n"
          + "\n"
          + "        ph.VALUE,\n"
          + "        LEAD(ph.VALUE) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_VALUE,\n"
          + "\n"
          + "        ph.LSL,\n"
          + "        LEAD(ph.LSL) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_LSL,\n"
          + "\n"
          + "        ph.USL,\n"
          + "        LEAD(ph.USL) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_USL,\n"
          + "\n"
          + "        LEAD(ph.CREATE_DTTS) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_CREATE_DTTS\n"
          + "\n"
          + "    FROM " + tableName + " ph\n"
          + "    CROSS JOIN p\n"
          + "    WHERE ph.MODULE_ID = p.module_id\n"
          + "      AND ph.PARAM_NAME = p.param_name\n"
          + "      AND ph.CREATE_DTTS >= p.start_dt\n"
          + "      AND ph.CREATE_DTTS <  p.end_dt\n"
          + "),\n"
          + "c AS (\n"
          + "    SELECT\n"
          + "        d.*,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS VALUE_CHANGED,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS LSL_CHANGED,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_USL, d.USL, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS USL_CHANGED\n"
          + "\n"
          + "    FROM d\n"
          + "    WHERE d.PREV_CREATE_DTTS IS NOT NULL\n"
          + "      AND d.PREV_VALUE IS NOT NULL\n"
          + "      AND d.PREV_LSL IS NOT NULL\n"
          + "      AND d.PREV_USL IS NOT NULL\n"
          + "      AND d.VALUE IS NOT NULL\n"
          + "      AND d.LSL IS NOT NULL\n"
          + "      AND d.USL IS NOT NULL\n"
          + "      AND (\n"
          + "             DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1\n"
          + "          OR DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1\n"
          + "          OR DECODE(d.PREV_USL, d.USL, 0, 1) = 1\n"
          + "      )\n"
          + ")\n"
          + "SELECT\n"
          + "    MODULE_ID,\n"
          + "    PARAM_NAME,\n"
          + "\n"
          + "    PREV_CREATE_DTTS AS PREVIOUS_CHANGE_TIME,\n"
          + "    CREATE_DTTS AS CURRENT_CHANGE_TIME,\n"
          + "\n"
          + "    CASE WHEN VALUE_CHANGED = 1 THEN PREV_VALUE END AS PREV_VALUE,\n"
          + "    CASE WHEN VALUE_CHANGED = 1 THEN VALUE END AS VALUE,\n"
          + "\n"
          + "    CASE WHEN LSL_CHANGED = 1 THEN PREV_LSL END AS PREV_LSL,\n"
          + "    CASE WHEN LSL_CHANGED = 1 THEN LSL END AS LSL,\n"
          + "\n"
          + "    CASE WHEN USL_CHANGED = 1 THEN PREV_USL END AS PREV_USL,\n"
          + "    CASE WHEN USL_CHANGED = 1 THEN USL END AS USL,\n"
          + "\n"
          + "    RTRIM(\n"
          + "          CASE WHEN VALUE_CHANGED = 1 THEN 'VALUE_CHANGED|' END\n"
          + "       || CASE WHEN LSL_CHANGED = 1 THEN 'LSL_CHANGED|' END\n"
          + "       || CASE WHEN USL_CHANGED = 1 THEN 'USL_CHANGED|' END,\n"
          + "       '|'\n"
          + "    ) AS CHANGE_TYPE\n"
          + "\n"
          + "FROM c\n"
          + "ORDER BY\n"
          + "    CURRENT_CHANGE_TIME";
    }

    private String buildSummarySql(String tableName) {
      return ""
          + "WITH p AS (\n"
          + "    SELECT\n"
          + "        TO_DATE(?, 'DDMMYYYY') AS start_dt,\n"
          + "        TO_DATE(?, 'DDMMYYYY') + 1 AS end_dt,\n"
          + "        TRIM(?) AS module_id,\n"
          + "        TRIM(?) AS param_name\n"
          + "    FROM dual\n"
          + "),\n"
          + "d AS (\n"
          + "    SELECT\n"
          + "        ph.ROWID AS rid,\n"
          + "        ph.MODULE_ID,\n"
          + "        ph.PARAM_NAME,\n"
          + "        ph.CREATE_DTTS,\n"
          + "\n"
          + "        ph.VALUE,\n"
          + "        LEAD(ph.VALUE) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_VALUE,\n"
          + "\n"
          + "        ph.LSL,\n"
          + "        LEAD(ph.LSL) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_LSL,\n"
          + "\n"
          + "        ph.USL,\n"
          + "        LEAD(ph.USL) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_USL,\n"
          + "\n"
          + "        LEAD(ph.CREATE_DTTS) OVER (\n"
          + "            PARTITION BY ph.MODULE_ID, ph.PARAM_NAME\n"
          + "            ORDER BY ph.CREATE_DTTS DESC, ph.ROWID DESC\n"
          + "        ) AS PREV_CREATE_DTTS\n"
          + "\n"
          + "    FROM " + tableName + " ph\n"
          + "    CROSS JOIN p\n"
          + "    WHERE ph.MODULE_ID = p.module_id\n"
          + "      AND ph.PARAM_NAME = p.param_name\n"
          + "      AND ph.CREATE_DTTS >= p.start_dt\n"
          + "      AND ph.CREATE_DTTS <  p.end_dt\n"
          + "),\n"
          + "c AS (\n"
          + "    SELECT\n"
          + "        d.MODULE_ID,\n"
          + "        d.PARAM_NAME,\n"
          + "        d.CREATE_DTTS,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS VALUE_CHANGED,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS LSL_CHANGED,\n"
          + "\n"
          + "        CASE\n"
          + "            WHEN DECODE(d.PREV_USL, d.USL, 0, 1) = 1\n"
          + "            THEN 1 ELSE 0\n"
          + "        END AS USL_CHANGED\n"
          + "\n"
          + "    FROM d\n"
          + "    WHERE d.PREV_CREATE_DTTS IS NOT NULL\n"
          + "      AND d.PREV_VALUE IS NOT NULL\n"
          + "      AND d.PREV_LSL IS NOT NULL\n"
          + "      AND d.PREV_USL IS NOT NULL\n"
          + "      AND d.VALUE IS NOT NULL\n"
          + "      AND d.LSL IS NOT NULL\n"
          + "      AND d.USL IS NOT NULL\n"
          + "      AND (\n"
          + "             DECODE(d.PREV_VALUE, d.VALUE, 0, 1) = 1\n"
          + "          OR DECODE(d.PREV_LSL, d.LSL, 0, 1) = 1\n"
          + "          OR DECODE(d.PREV_USL, d.USL, 0, 1) = 1\n"
          + "      )\n"
          + ")\n"
          + "SELECT\n"
          + "    MODULE_ID,\n"
          + "    PARAM_NAME,\n"
          + "\n"
          + "    MIN(CREATE_DTTS) AS FIRST_CHANGE_TIME,\n"
          + "    MAX(CREATE_DTTS) AS LAST_CHANGE_TIME,\n"
          + "\n"
          + "    SUM(VALUE_CHANGED) AS VALUE_CHANGE_COUNT,\n"
          + "    SUM(LSL_CHANGED) AS LSL_CHANGE_COUNT,\n"
          + "    SUM(USL_CHANGED) AS USL_CHANGE_COUNT,\n"
          + "\n"
          + "    COUNT(*) AS TOTAL_CHANGE_POINTS,\n"
          + "\n"
          + "    RTRIM(\n"
          + "          CASE WHEN SUM(VALUE_CHANGED) > 0 THEN 'VALUE_CHANGED|' END\n"
          + "       || CASE WHEN SUM(LSL_CHANGED) > 0 THEN 'LSL_CHANGED|' END\n"
          + "       || CASE WHEN SUM(USL_CHANGED) > 0 THEN 'USL_CHANGED|' END,\n"
          + "       '|'\n"
          + "    ) AS CHANGE_TYPE\n"
          + "\n"
          + "FROM c\n"
          + "GROUP BY\n"
          + "    MODULE_ID,\n"
          + "    PARAM_NAME";
    }

    private String buildMainInsertSql(String targetTable) {
      return "INSERT INTO " + targetTable + " (run_dtts, source, module_id, param_name, previous_change_time, current_change_time, prev_value, value, prev_lsl, lsl, prev_usl, usl, change_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    }

    private String buildSummaryInsertSql(String targetTable) {
      return "INSERT INTO " + targetTable + " (run_dtts, source, module_id, param_name, first_change_time, last_change_time, value_change_count, lsl_change_count, usl_change_count, total_change_points, change_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    }

    private String buildTrendInsertSql(String targetTable) {
      return "INSERT INTO " + targetTable + " (run_dtts, source, module_id, param_name, previous_change_time, current_change_time, prev_value, value, prev_lsl, lsl, prev_usl, usl, change_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    }

    private void bindDateRange(PreparedStatement ps, DateRange range) throws SQLException {
      ps.setString(1, range.startDate);
      ps.setString(2, range.endDate);
    }

    private void bindDateRangeModuleParam(PreparedStatement ps, DateRange range, String moduleId,
        String paramName) throws SQLException {
      bindDateRange(ps, range);
      ps.setString(3, moduleId);
      ps.setString(4, paramName);
    }
  }

  private static Connection openConnection(ServletContext context, String urlKey, String urlEnv,
      String userKey, String userEnv, String passwordKey, String passwordEnv) throws SQLException {
    String url = requireConfig(context, urlKey, urlEnv);
    String user = requireConfig(context, userKey, userEnv);
    String password = requireConfig(context, passwordKey, passwordEnv);
    return DriverManager.getConnection(url, user, password);
  }

  private static String requireConfig(ServletContext context, String key, String envKey) {
    String value = getConfig(context, key, envKey, null);
    if (value == null || value.length() == 0) {
      throw new IllegalStateException("Missing required configuration: " + key + " or " + envKey);
    }
    return value;
  }

  private static String getConfig(ServletContext context, String key, String envKey, String fallback) {
    String value = context.getInitParameter(key);
    if (value == null || value.length() == 0) value = System.getProperty(key);
    if (value == null || value.length() == 0) value = System.getenv(envKey);
    return value == null || value.length() == 0 ? fallback : value;
  }

  private static long getLong(ServletContext context, String key, long fallback) {
    String value = getConfig(context, key, key.replace('.', '_').toUpperCase(), null);
    if (value == null) return fallback;
    try {
      return Long.parseLong(value);
    } catch (NumberFormatException e) {
      return fallback;
    }
  }

  private static DateRange lastHourRange() {
    Calendar start = Calendar.getInstance();
    start.add(Calendar.HOUR_OF_DAY, -1);
    Calendar end = Calendar.getInstance();
    SimpleDateFormat format = new SimpleDateFormat("ddMMyyyy");
    return new DateRange(format.format(start.getTime()), format.format(end.getTime()));
  }

  private static final class DateRange {
    final String startDate;
    final String endDate;

    DateRange(String startDate, String endDate) {
      this.startDate = startDate;
      this.endDate = endDate;
    }
  }

  private static final class CopyResult {
    int mainRows;
    int summaryRows;
    int trendRows;
    final Set<ModuleParameter> keys = new LinkedHashSet<ModuleParameter>();

    void addKey(String moduleId, String paramName) {
      keys.add(new ModuleParameter(moduleId, paramName));
    }

    void add(CopyResult other) {
      mainRows += other.mainRows;
      summaryRows += other.summaryRows;
      trendRows += other.trendRows;
    }

    public String toString() {
      return "main=" + mainRows + ", summary=" + summaryRows + ", trend=" + trendRows;
    }
  }

  private static final class ModuleParameter {
    final String moduleId;
    final String paramName;

    ModuleParameter(String moduleId, String paramName) {
      this.moduleId = moduleId;
      this.paramName = paramName;
    }

    public boolean equals(Object other) {
      if (!(other instanceof ModuleParameter)) return false;
      ModuleParameter that = (ModuleParameter) other;
      return same(moduleId, that.moduleId) && same(paramName, that.paramName);
    }

    public int hashCode() {
      int result = moduleId == null ? 0 : moduleId.hashCode();
      result = 31 * result + (paramName == null ? 0 : paramName.hashCode());
      return result;
    }

    private static boolean same(String left, String right) {
      return left == null ? right == null : left.equals(right);
    }
  }

  private static void rollback(Connection connection) {
    if (connection != null) {
      try {
        connection.rollback();
      } catch (SQLException ignored) {
      }
    }
  }

}
