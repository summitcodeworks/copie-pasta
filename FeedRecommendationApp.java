import java.sql.*;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;

public class FeedRecommendationApp {

    // =========================================================
    // CONFIG
    // =========================================================
    private static final String DB_URL = "jdbc:postgresql://localhost:5432/feed_db";
    private static final String DB_USER = "postgres";
    private static final String DB_PASSWORD = "postgres";

    private static final int DEFAULT_PAGE_SIZE = 10;
    private static final int CANDIDATE_POOL_SIZE = 300;
    private static final int RECENT_IMPRESSION_DAYS = 7;
    private static final int OLD_POST_HARD_LIMIT_DAYS = 30;

    // =========================================================
    // EVENT WEIGHTS
    // =========================================================
    private static final Map<String, Double> EVENT_WEIGHTS = new HashMap<>();

    static {
        EVENT_WEIGHTS.put("view", 0.5);
        EVENT_WEIGHTS.put("click", 1.5);
        EVENT_WEIGHTS.put("like", 3.0);
        EVENT_WEIGHTS.put("comment", 4.0);
        EVENT_WEIGHTS.put("share", 5.0);
        EVENT_WEIGHTS.put("save", 4.5);
        EVENT_WEIGHTS.put("skip", -2.0);
    }

    // =========================================================
    // MAIN
    // =========================================================
    public static void main(String[] args) {
        FeedRecommendationApp app = new FeedRecommendationApp();

        try {
            app.loadDriver();
            app.initDatabase();
            app.seedDemoData();

            long userId = 1L;
            String sessionId = UUID.randomUUID().toString();

            System.out.println("\n================ FIRST FEED PAGE ================\n");
            FeedPage firstPage = app.getFeedPage(userId, DEFAULT_PAGE_SIZE, null, sessionId);
            app.printFeedPage(firstPage);

            if (!firstPage.posts.isEmpty()) {
                // Simulate user behavior on first few posts
                Post p1 = firstPage.posts.get(0);
                app.logEvent(userId, p1.id, "view", 12.0);
                app.logEvent(userId, p1.id, "like", 0.0);

                if (firstPage.posts.size() > 1) {
                    Post p2 = firstPage.posts.get(1);
                    app.logEvent(userId, p2.id, "view", 18.0);
                    app.logEvent(userId, p2.id, "comment", 0.0);
                }

                if (firstPage.posts.size() > 2) {
                    Post p3 = firstPage.posts.get(2);
                    app.logEvent(userId, p3.id, "skip", 1.0);
                }
            }

            System.out.println("\n================ SECOND FEED PAGE ================\n");
            FeedPage secondPage = app.getFeedPage(
                    userId,
                    DEFAULT_PAGE_SIZE,
                    firstPage.nextCursorCreatedAt,
                    sessionId
            );
            app.printFeedPage(secondPage);

            System.out.println("\n================ USER INTEREST PROFILE ================\n");
            Map<String, Double> interestProfile = app.getUserCategoryScores(userId);
            for (Map.Entry<String, Double> entry : interestProfile.entrySet()) {
                System.out.printf("%-15s -> %.2f%n", entry.getKey(), entry.getValue());
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // =========================================================
    // JDBC DRIVER
    // =========================================================
    public void loadDriver() throws ClassNotFoundException {
        Class.forName("org.postgresql.Driver");
    }

    // =========================================================
    // DB CONNECTION
    // =========================================================
    private Connection getConnection() throws SQLException {
        return DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
    }

    // =========================================================
    // INIT DATABASE
    // =========================================================
    public void initDatabase() throws SQLException {
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement()) {

            stmt.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username VARCHAR(100) NOT NULL UNIQUE,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """);

            stmt.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id BIGSERIAL PRIMARY KEY,
                    author_id BIGINT NOT NULL REFERENCES users(id),
                    category VARCHAR(100),
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    like_count INT NOT NULL DEFAULT 0,
                    comment_count INT NOT NULL DEFAULT 0,
                    share_count INT NOT NULL DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                )
            """);

            stmt.execute("""
                CREATE TABLE IF NOT EXISTS user_post_events (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    post_id BIGINT NOT NULL REFERENCES posts(id),
                    event_type VARCHAR(30) NOT NULL,
                    dwell_time_seconds DOUBLE PRECISION DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """);

            stmt.execute("""
                CREATE TABLE IF NOT EXISTS user_post_impressions (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    post_id BIGINT NOT NULL REFERENCES posts(id),
                    shown_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    session_id VARCHAR(100)
                )
            """);

            stmt.execute("""
                CREATE TABLE IF NOT EXISTS user_interest_profile (
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    category VARCHAR(100) NOT NULL,
                    score DOUBLE PRECISION NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_id, category)
                )
            """);

            stmt.execute("CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_posts_author_id ON posts(author_id)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_posts_category ON posts(category)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_user_post_events_user_created ON user_post_events(user_id, created_at DESC)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_user_post_events_post ON user_post_events(post_id)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_user_post_impressions_user_post ON user_post_impressions(user_id, post_id)");
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_user_post_impressions_user_shown_at ON user_post_impressions(user_id, shown_at DESC)");
        }
    }

    // =========================================================
    // SEED DEMO DATA
    // =========================================================
    public void seedDemoData() throws SQLException {
        try (Connection conn = getConnection()) {
            conn.setAutoCommit(false);

            long existingUsers = countRows(conn, "users");
            if (existingUsers > 0) {
                conn.commit();
                return;
            }

            List<Long> userIds = new ArrayList<>();
            userIds.add(insertUser(conn, "viewer_user"));     // id 1 intended consumer
            userIds.add(insertUser(conn, "author_tech"));
            userIds.add(insertUser(conn, "author_sports"));
            userIds.add(insertUser(conn, "author_music"));
            userIds.add(insertUser(conn, "author_food"));
            userIds.add(insertUser(conn, "author_travel"));

            // Fresh posts
            insertPost(conn, userIds.get(1), "technology", "AI recommendation systems are evolving fast", 2, 55, 12, 8);
            insertPost(conn, userIds.get(1), "technology", "Spring Boot 3 performance tuning tricks", 4, 40, 7, 5);
            insertPost(conn, userIds.get(1), "technology", "Java virtual threads practical guide", 8, 70, 15, 9);

            insertPost(conn, userIds.get(2), "sports", "Top cricket moments from the latest season", 1, 100, 24, 14);
            insertPost(conn, userIds.get(2), "sports", "Why fast bowlers dominate in night matches", 3, 35, 6, 4);
            insertPost(conn, userIds.get(2), "sports", "Football press resistance explained simply", 5, 30, 5, 2);

            insertPost(conn, userIds.get(3), "music", "Best indie albums this month", 2, 28, 4, 3);
            insertPost(conn, userIds.get(3), "music", "How producers layer vocals for depth", 6, 32, 6, 4);
            insertPost(conn, userIds.get(3), "music", "Beginner guitar chord transitions", 10, 50, 9, 7);

            insertPost(conn, userIds.get(4), "food", "5 easy high-protein breakfast ideas", 1, 65, 8, 6);
            insertPost(conn, userIds.get(4), "food", "Street food spots worth trying this weekend", 7, 120, 20, 12);
            insertPost(conn, userIds.get(4), "food", "How to make restaurant-style paneer at home", 12, 80, 11, 10);

            insertPost(conn, userIds.get(5), "travel", "Underrated hill destinations for summer trips", 2, 60, 10, 7);
            insertPost(conn, userIds.get(5), "travel", "Budget backpacking essentials for beginners", 5, 42, 6, 5);
            insertPost(conn, userIds.get(5), "travel", "3-day itinerary for a quick coastal getaway", 9, 38, 7, 4);

            // Some older posts
            insertPost(conn, userIds.get(1), "technology", "Old Java 8 tricks that still matter", 18, 25, 2, 1);
            insertPost(conn, userIds.get(2), "sports", "Classic ODI run chases every fan remembers", 22, 18, 2, 1);
            insertPost(conn, userIds.get(3), "music", "Retro synth sound design overview", 27, 15, 1, 1);
            insertPost(conn, userIds.get(4), "food", "Traditional recipes from festival season", 33, 14, 1, 0); // beyond hard filter
            insertPost(conn, userIds.get(5), "travel", "A very old solo travel checklist", 40, 11, 1, 0); // beyond hard filter

            // Seed a few historical user events so algorithm has preference data
            long viewerId = userIds.get(0);
            List<Post> techPosts = getPostsByCategory(conn, "technology");
            List<Post> sportsPosts = getPostsByCategory(conn, "sports");
            List<Post> foodPosts = getPostsByCategory(conn, "food");

            if (!techPosts.isEmpty()) {
                insertEvent(conn, viewerId, techPosts.get(0).id, "view", 15.0, 2);
                insertEvent(conn, viewerId, techPosts.get(0).id, "like", 0.0, 2);
                insertEvent(conn, viewerId, techPosts.get(1).id, "view", 20.0, 1);
                insertEvent(conn, viewerId, techPosts.get(1).id, "save", 0.0, 1);
            }
            if (!sportsPosts.isEmpty()) {
                insertEvent(conn, viewerId, sportsPosts.get(0).id, "view", 14.0, 3);
                insertEvent(conn, viewerId, sportsPosts.get(0).id, "comment", 0.0, 3);
            }
            if (!foodPosts.isEmpty()) {
                insertEvent(conn, viewerId, foodPosts.get(0).id, "view", 3.0, 2);
                insertEvent(conn, viewerId, foodPosts.get(0).id, "skip", 1.0, 2);
            }

            conn.commit();
        }
    }

    // =========================================================
    // FEED PAGE API
    // cursorCreatedAt:
    //   null  -> first page
    //   value -> next page
    // =========================================================
    public FeedPage getFeedPage(long userId, int pageSize, LocalDateTime cursorCreatedAt, String sessionId) throws SQLException {
        updateUserInterestProfile(userId);

        Map<String, Double> categoryScores = getUserCategoryScores(userId);
        RecentContext recentContext = getRecentAuthorsAndCategories(userId, 20);
        List<Post> candidates = getCandidatePosts(userId, CANDIDATE_POOL_SIZE, cursorCreatedAt);

        List<ScoredPost> scoredPosts = new ArrayList<>();
        for (Post post : candidates) {
            if (!hardFilterPost(post)) {
                continue;
            }

            double score = scorePost(
                    post,
                    categoryScores,
                    recentContext.recentAuthors,
                    recentContext.recentCategories
            );

            scoredPosts.add(new ScoredPost(post, score));
        }

        scoredPosts.sort((a, b) -> {
            int scoreCompare = Double.compare(b.score, a.score);
            if (scoreCompare != 0) return scoreCompare;
            return b.post.createdAt.compareTo(a.post.createdAt);
        });

        List<Post> pagePosts = diversifyFeed(scoredPosts, pageSize);
        saveImpressions(userId, pagePosts, sessionId);

        LocalDateTime nextCursor = null;
        boolean hasMore = false;
        if (!pagePosts.isEmpty()) {
            nextCursor = pagePosts.get(pagePosts.size() - 1).createdAt;
            hasMore = candidates.size() > pagePosts.size();
        }

        return new FeedPage(pagePosts, nextCursor, hasMore);
    }

    // =========================================================
    // UPDATE USER INTEREST PROFILE
    // =========================================================
    public void updateUserInterestProfile(long userId) throws SQLException {
        String sql = """
            SELECT p.category, e.event_type, e.dwell_time_seconds, e.created_at
            FROM user_post_events e
            JOIN posts p ON p.id = e.post_id
            WHERE e.user_id = ?
              AND e.created_at >= NOW() - INTERVAL '30 days'
        """;

        Map<String, Double> scores = new HashMap<>();

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setLong(1, userId);

            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    String category = rs.getString("category");
                    String eventType = rs.getString("event_type");
                    double dwell = rs.getDouble("dwell_time_seconds");
                    Timestamp ts = rs.getTimestamp("created_at");

                    if (category == null || category.isBlank()) continue;

                    double base = EVENT_WEIGHTS.getOrDefault(eventType, 0.0);
                    double dwellBonus = Math.min(dwell / 10.0, 3.0);

                    // optional recency factor on events
                    double daysAgo = 0.0;
                    if (ts != null) {
                        LocalDateTime eventTime = ts.toLocalDateTime();
                        daysAgo = Duration.between(eventTime, LocalDateTime.now()).toHours() / 24.0;
                    }
                    double recencyFactor = Math.exp(-daysAgo / 15.0);

                    double total = (base + dwellBonus) * recencyFactor;
                    scores.put(category, scores.getOrDefault(category, 0.0) + total);
                }
            }
        }

        String upsert = """
            INSERT INTO user_interest_profile (user_id, category, score, updated_at)
            VALUES (?, ?, ?, NOW())
            ON CONFLICT (user_id, category)
            DO UPDATE SET score = EXCLUDED.score, updated_at = NOW()
        """;

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(upsert)) {

            for (Map.Entry<String, Double> entry : scores.entrySet()) {
                ps.setLong(1, userId);
                ps.setString(2, entry.getKey());
                ps.setDouble(3, entry.getValue());
                ps.addBatch();
            }

            ps.executeBatch();
        }
    }

    // =========================================================
    // GET USER CATEGORY SCORES
    // =========================================================
    public Map<String, Double> getUserCategoryScores(long userId) throws SQLException {
        String sql = """
            SELECT category, score
            FROM user_interest_profile
            WHERE user_id = ?
        """;

        Map<String, Double> result = new HashMap<>();

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setLong(1, userId);

            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    result.put(rs.getString("category"), rs.getDouble("score"));
                }
            }
        }

        return result;
    }

    // =========================================================
    // RECENT CONTEXT
    // =========================================================
    public RecentContext getRecentAuthorsAndCategories(long userId, int limit) throws SQLException {
        String sql = """
            SELECT p.author_id, p.category
            FROM user_post_impressions i
            JOIN posts p ON p.id = i.post_id
            WHERE i.user_id = ?
            ORDER BY i.shown_at DESC
            LIMIT ?
        """;

        List<Long> authors = new ArrayList<>();
        List<String> categories = new ArrayList<>();

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            ps.setLong(1, userId);
            ps.setInt(2, limit);

            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    authors.add(rs.getLong("author_id"));
                    String category = rs.getString("category");
                    if (category != null && !category.isBlank()) {
                        categories.add(category);
                    }
                }
            }
        }

        return new RecentContext(authors, categories);
    }

    // =========================================================
    // CANDIDATE POSTS
    // Excludes recently shown posts
    // Supports cursor for infinite scroll
    // =========================================================
    public List<Post> getCandidatePosts(long userId, int limit, LocalDateTime cursorCreatedAt) throws SQLException {
        StringBuilder sql = new StringBuilder("""
            SELECT p.id, p.author_id, p.category, p.content, p.created_at,
                   p.like_count, p.comment_count, p.share_count, p.is_active
            FROM posts p
            WHERE p.is_active = TRUE
              AND p.id NOT IN (
                  SELECT i.post_id
                  FROM user_post_impressions i
                  WHERE i.user_id = ?
                    AND i.shown_at >= NOW() - INTERVAL '7 days'
              )
        """);

        if (cursorCreatedAt != null) {
            sql.append(" AND p.created_at < ? ");
        }

        sql.append(" ORDER BY p.created_at DESC LIMIT ? ");

        List<Post> posts = new ArrayList<>();

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql.toString())) {

            int index = 1;
            ps.setLong(index++, userId);

            if (cursorCreatedAt != null) {
                ps.setTimestamp(index++, Timestamp.valueOf(cursorCreatedAt));
            }

            ps.setInt(index, limit);

            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    posts.add(mapPost(rs));
                }
            }
        }

        return posts;
    }

    // =========================================================
    // SCORING
    // =========================================================
    public double scorePost(Post post,
                            Map<String, Double> categoryScores,
                            List<Long> recentAuthors,
                            List<String> recentCategories) {
        double interest = categoryScores.getOrDefault(post.category, 0.0);
        double freshness = freshnessScore(post);
        double popularity = engagementScore(post);
        double repeatPenalty = repetitionPenalty(post, recentAuthors, recentCategories);
        double oldPenalty = oldPostPenalty(post);

        return interest + freshness + popularity - repeatPenalty - oldPenalty;
    }

    public double freshnessScore(Post post) {
        double ageHours = hoursSince(post.createdAt);
        return 10.0 * Math.exp(-ageHours / 48.0);
    }

    public double engagementScore(Post post) {
        return (post.likeCount * 0.05)
                + (post.commentCount * 0.10)
                + (post.shareCount * 0.15);
    }

    public double repetitionPenalty(Post post, List<Long> recentAuthors, List<String> recentCategories) {
        double penalty = 0.0;

        long sameAuthorCount = recentAuthors.stream().filter(a -> a.equals(post.authorId)).count();
        long sameCategoryCount = recentCategories.stream()
                .filter(c -> Objects.equals(c, post.category))
                .count();

        if (sameAuthorCount >= 2) {
            penalty += 3.0;
        }

        if (post.category != null && sameCategoryCount >= 3) {
            penalty += 2.5;
        }

        return penalty;
    }

    public double oldPostPenalty(Post post) {
        long ageDays = Duration.between(post.createdAt, LocalDateTime.now()).toDays();

        if (ageDays <= 3) return 0.0;
        if (ageDays <= 7) return 1.0;
        if (ageDays <= 15) return 3.0;
        return 6.0;
    }

    public boolean hardFilterPost(Post post) {
        long ageDays = Duration.between(post.createdAt, LocalDateTime.now()).toDays();
        return ageDays <= OLD_POST_HARD_LIMIT_DAYS;
    }

    public double hoursSince(LocalDateTime createdAt) {
        return Math.max(Duration.between(createdAt, LocalDateTime.now()).toMinutes() / 60.0, 0.0);
    }

    // =========================================================
    // DIVERSITY
    // =========================================================
    public List<Post> diversifyFeed(List<ScoredPost> scoredPosts, int pageSize) {
        List<Post> result = new ArrayList<>();
        Map<Long, Integer> authorUsage = new HashMap<>();
        Map<String, Integer> categoryUsage = new HashMap<>();

        for (ScoredPost scored : scoredPosts) {
            Post post = scored.post;

            int authorCount = authorUsage.getOrDefault(post.authorId, 0);
            int categoryCount = categoryUsage.getOrDefault(post.category, 0);

            if (authorCount >= 2) {
                continue;
            }

            if (post.category != null && categoryCount >= 3) {
                continue;
            }

            result.add(post);
            authorUsage.put(post.authorId, authorCount + 1);

            if (post.category != null) {
                categoryUsage.put(post.category, categoryCount + 1);
            }

            if (result.size() >= pageSize) {
                break;
            }
        }

        return result;
    }

    // =========================================================
    // IMPRESSION LOGGING
    // =========================================================
    public void saveImpressions(long userId, List<Post> posts, String sessionId) throws SQLException {
        if (posts == null || posts.isEmpty()) return;

        String sql = """
            INSERT INTO user_post_impressions (user_id, post_id, shown_at, session_id)
            VALUES (?, ?, NOW(), ?)
        """;

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {

            for (Post post : posts) {
                ps.setLong(1, userId);
                ps.setLong(2, post.id);
                ps.setString(3, sessionId);
                ps.addBatch();
            }

            ps.executeBatch();
        }
    }

    // =========================================================
    // EVENT LOGGING
    // Also updates aggregate counts on posts table
    // =========================================================
    public void logEvent(long userId, long postId, String eventType, double dwellTimeSeconds) throws SQLException {
        try (Connection conn = getConnection()) {
            conn.setAutoCommit(false);

            String insertEvent = """
                INSERT INTO user_post_events (user_id, post_id, event_type, dwell_time_seconds, created_at)
                VALUES (?, ?, ?, ?, NOW())
            """;

            try (PreparedStatement ps = conn.prepareStatement(insertEvent)) {
                ps.setLong(1, userId);
                ps.setLong(2, postId);
                ps.setString(3, eventType);
                ps.setDouble(4, dwellTimeSeconds);
                ps.executeUpdate();
            }

            if ("like".equalsIgnoreCase(eventType)) {
                incrementPostMetric(conn, postId, "like_count");
            } else if ("comment".equalsIgnoreCase(eventType)) {
                incrementPostMetric(conn, postId, "comment_count");
            } else if ("share".equalsIgnoreCase(eventType)) {
                incrementPostMetric(conn, postId, "share_count");
            }

            conn.commit();
        }
    }

    private void incrementPostMetric(Connection conn, long postId, String columnName) throws SQLException {
        String sql = "UPDATE posts SET " + columnName + " = " + columnName + " + 1 WHERE id = ?";
        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, postId);
            ps.executeUpdate();
        }
    }

    // =========================================================
    // HELPERS
    // =========================================================
    private Post mapPost(ResultSet rs) throws SQLException {
        Post post = new Post();
        post.id = rs.getLong("id");
        post.authorId = rs.getLong("author_id");
        post.category = rs.getString("category");
        post.content = rs.getString("content");
        post.createdAt = rs.getTimestamp("created_at").toLocalDateTime();
        post.likeCount = rs.getInt("like_count");
        post.commentCount = rs.getInt("comment_count");
        post.shareCount = rs.getInt("share_count");
        post.isActive = rs.getBoolean("is_active");
        return post;
    }

    private long countRows(Connection conn, String table) throws SQLException {
        try (Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM " + table)) {
            rs.next();
            return rs.getLong(1);
        }
    }

    private long insertUser(Connection conn, String username) throws SQLException {
        String sql = """
            INSERT INTO users (username, created_at)
            VALUES (?, NOW())
            RETURNING id
        """;

        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, username);
            try (ResultSet rs = ps.executeQuery()) {
                rs.next();
                return rs.getLong("id");
            }
        }
    }

    private long insertPost(Connection conn,
                            long authorId,
                            String category,
                            String content,
                            int ageDays,
                            int likeCount,
                            int commentCount,
                            int shareCount) throws SQLException {
        String sql = """
            INSERT INTO posts (
                author_id, category, content, created_at,
                like_count, comment_count, share_count, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, TRUE)
            RETURNING id
        """;

        LocalDateTime createdAt = LocalDateTime.now().minusDays(ageDays);

        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, authorId);
            ps.setString(2, category);
            ps.setString(3, content);
            ps.setTimestamp(4, Timestamp.valueOf(createdAt));
            ps.setInt(5, likeCount);
            ps.setInt(6, commentCount);
            ps.setInt(7, shareCount);

            try (ResultSet rs = ps.executeQuery()) {
                rs.next();
                return rs.getLong("id");
            }
        }
    }

    private void insertEvent(Connection conn,
                             long userId,
                             long postId,
                             String eventType,
                             double dwellTime,
                             int daysAgo) throws SQLException {
        String sql = """
            INSERT INTO user_post_events (
                user_id, post_id, event_type, dwell_time_seconds, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """;

        LocalDateTime createdAt = LocalDateTime.now().minusDays(daysAgo);

        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setLong(1, userId);
            ps.setLong(2, postId);
            ps.setString(3, eventType);
            ps.setDouble(4, dwellTime);
            ps.setTimestamp(5, Timestamp.valueOf(createdAt));
            ps.executeUpdate();
        }
    }

    private List<Post> getPostsByCategory(Connection conn, String category) throws SQLException {
        String sql = """
            SELECT id, author_id, category, content, created_at,
                   like_count, comment_count, share_count, is_active
            FROM posts
            WHERE category = ?
            ORDER BY created_at DESC
        """;

        List<Post> posts = new ArrayList<>();

        try (PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, category);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    posts.add(mapPost(rs));
                }
            }
        }

        return posts;
    }

    public void printFeedPage(FeedPage page) {
        System.out.println("Posts count: " + page.posts.size());
        System.out.println("Has more   : " + page.hasMore);
        System.out.println("Next cursor: " + page.nextCursorCreatedAt);
        System.out.println();

        int index = 1;
        for (Post post : page.posts) {
            System.out.println("[" + index++ + "] Post ID      : " + post.id);
            System.out.println("    Author ID   : " + post.authorId);
            System.out.println("    Category    : " + post.category);
            System.out.println("    Created At  : " + post.createdAt);
            System.out.println("    Likes       : " + post.likeCount);
            System.out.println("    Comments    : " + post.commentCount);
            System.out.println("    Shares      : " + post.shareCount);
            System.out.println("    Content     : " + post.content);
            System.out.println();
        }
    }

    // =========================================================
    // MODELS
    // =========================================================
    static class Post {
        long id;
        long authorId;
        String category;
        String content;
        LocalDateTime createdAt;
        int likeCount;
        int commentCount;
        int shareCount;
        boolean isActive;
    }

    static class ScoredPost {
        Post post;
        double score;

        ScoredPost(Post post, double score) {
            this.post = post;
            this.score = score;
        }
    }

    static class RecentContext {
        List<Long> recentAuthors;
        List<String> recentCategories;

        RecentContext(List<Long> recentAuthors, List<String> recentCategories) {
            this.recentAuthors = recentAuthors;
            this.recentCategories = recentCategories;
        }
    }

    static class FeedPage {
        List<Post> posts;
        LocalDateTime nextCursorCreatedAt;
        boolean hasMore;

        FeedPage(List<Post> posts, LocalDateTime nextCursorCreatedAt, boolean hasMore) {
            this.posts = posts;
            this.nextCursorCreatedAt = nextCursorCreatedAt;
            this.hasMore = hasMore;
        }
    }
}
