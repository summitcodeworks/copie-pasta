import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java 6 compatible example that reads an image as bytes, Base64-encodes it,
 * gives it a timestamp-based name, and sends it to an API with other fields.
 *
 * No third-party libraries are required.
 */
public final class ImageBase64ApiClient {
    private static final String UTF_8 = "UTF-8";
    private static final String SAMPLE_API_USERNAME = "sampleUser";
    private static final String SAMPLE_API_PASSWORD = "samplePassword";
    private static final char[] BASE64_CHARACTERS =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".toCharArray();

    private ImageBase64ApiClient() {
        // Utility class.
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java ImageBase64ApiClient <api-url> <image-file>");
            return;
        }

        File imageFile = new File(args[1]);

        Map<String, String> otherParameters = new LinkedHashMap<String, String>();
        otherParameters.put("userId", "12345");
        otherParameters.put("description", "Image uploaded from Java 6");

        try {
            String apiKey = createBasicAuthValue(SAMPLE_API_USERNAME, SAMPLE_API_PASSWORD);
            ApiResponse response = uploadImage(args[0], imageFile, otherParameters, apiKey);
            System.out.println("HTTP status: " + response.getStatusCode());
            System.out.println("Response: " + response.getBody());
        } catch (Exception exception) {
            System.err.println("Upload failed: " + exception.getMessage());
        }
    }

    public static ApiResponse uploadImage(String apiUrl, File imageFile,
            Map<String, String> otherParameters, String apiKey) throws IOException {
        validateImageFile(imageFile);

        byte[] imageBytes = readFileToByteArray(imageFile);
        String encodedImage = encodeBase64(imageBytes);
        String timestampFileName = createTimestampFileName(imageFile);

        Map<String, String> requestParameters = new LinkedHashMap<String, String>();
        if (otherParameters != null) {
            requestParameters.putAll(otherParameters);
        }
        requestParameters.put("fileName", timestampFileName);
        requestParameters.put("imageBase64", encodedImage);

        return postForm(apiUrl, requestParameters, apiKey);
    }

    public static String createBasicAuthValue(String username, String password)
            throws IOException {
        String credentials = username + ":" + password;
        return "Basic " + encodeBase64(credentials.getBytes(UTF_8));
    }

    public static byte[] readFileToByteArray(File file) throws IOException {
        InputStream input = new BufferedInputStream(new FileInputStream(file));
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        try {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = input.read(buffer)) != -1) {
                output.write(buffer, 0, bytesRead);
            }
            return output.toByteArray();
        } finally {
            try {
                input.close();
            } finally {
                output.close();
            }
        }
    }

    public static String createTimestampFileName(File file) {
        String extension = getExtension(file.getName());
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS").format(new Date());
        return timestamp + extension;
    }

    public static String encodeBase64(byte[] data) {
        StringBuilder encoded = new StringBuilder(((data.length + 2) / 3) * 4);

        for (int index = 0; index < data.length; index += 3) {
            int first = data[index] & 0xFF;
            int second = index + 1 < data.length ? data[index + 1] & 0xFF : 0;
            int third = index + 2 < data.length ? data[index + 2] & 0xFF : 0;

            encoded.append(BASE64_CHARACTERS[first >>> 2]);
            encoded.append(BASE64_CHARACTERS[((first & 0x03) << 4) | (second >>> 4)]);
            encoded.append(index + 1 < data.length
                    ? BASE64_CHARACTERS[((second & 0x0F) << 2) | (third >>> 6)] : '=');
            encoded.append(index + 2 < data.length ? BASE64_CHARACTERS[third & 0x3F] : '=');
        }

        return encoded.toString();
    }

    private static ApiResponse postForm(String apiUrl, Map<String, String> parameters,
            String apiKey) throws IOException {
        byte[] requestBody = createFormBody(parameters).getBytes(UTF_8);
        HttpURLConnection connection = (HttpURLConnection) new URL(apiUrl).openConnection();

        try {
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(15000);
            connection.setReadTimeout(30000);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type",
                    "application/x-www-form-urlencoded; charset=UTF-8");
            connection.setRequestProperty("Accept", "application/json, text/plain, */*");
            connection.setRequestProperty("apiKey", apiKey);
            connection.setFixedLengthStreamingMode(requestBody.length);

            OutputStream output = connection.getOutputStream();
            try {
                output.write(requestBody);
                output.flush();
            } finally {
                output.close();
            }

            int statusCode = connection.getResponseCode();
            InputStream responseStream = statusCode >= 200 && statusCode < 400
                    ? connection.getInputStream() : connection.getErrorStream();
            String responseBody = readResponse(responseStream);
            return new ApiResponse(statusCode, responseBody);
        } finally {
            connection.disconnect();
        }
    }

    private static String createFormBody(Map<String, String> parameters)
            throws IOException {
        StringBuilder body = new StringBuilder();

        for (Map.Entry<String, String> parameter : parameters.entrySet()) {
            if (body.length() > 0) {
                body.append('&');
            }
            body.append(URLEncoder.encode(parameter.getKey(), UTF_8));
            body.append('=');
            body.append(URLEncoder.encode(parameter.getValue(), UTF_8));
        }

        return body.toString();
    }

    private static String readResponse(InputStream input) throws IOException {
        if (input == null) {
            return "";
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(input, UTF_8));
        StringBuilder response = new StringBuilder();

        try {
            String line;
            while ((line = reader.readLine()) != null) {
                if (response.length() > 0) {
                    response.append('\n');
                }
                response.append(line);
            }
            return response.toString();
        } finally {
            reader.close();
        }
    }

    private static void validateImageFile(File file) throws IOException {
        if (file == null || !file.isFile()) {
            throw new IOException("Image file does not exist: " + file);
        }
        if (!file.canRead()) {
            throw new IOException("Image file cannot be read: " + file.getAbsolutePath());
        }
        if (getExtension(file.getName()).length() == 0) {
            throw new IOException("Image file must have an extension: " + file.getName());
        }
    }

    private static String getExtension(String fileName) {
        int dotPosition = fileName.lastIndexOf('.');
        if (dotPosition <= 0 || dotPosition == fileName.length() - 1) {
            return "";
        }
        return fileName.substring(dotPosition).toLowerCase();
    }

    public static final class ApiResponse {
        private final int statusCode;
        private final String body;

        public ApiResponse(int statusCode, String body) {
            this.statusCode = statusCode;
            this.body = body;
        }

        public int getStatusCode() {
            return statusCode;
        }

        public String getBody() {
            return body;
        }
    }
}
