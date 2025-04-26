// API configuration
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:9091"
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:9091"

// The environment variables will be used in production
// Local development will fall back to localhost if the variables aren't set
