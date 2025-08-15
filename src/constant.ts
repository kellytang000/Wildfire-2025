import { join } from "path";

export const PYRECAST_BASE_URL = "https://data.pyrecast.org/fire_spread_forecast/";

export const FIRES_URL = "https://data.pyrecast.org/fire_detections/active-fires/active-fires.csv";

export const JWT_SECRET = "bzEouAmeHVpYo5uONIHVT5UfdmpqtniKF9WwrXpeEEA=";

export const SCRIPT_PATH = join(process.cwd(), "src", "script", "squa");
