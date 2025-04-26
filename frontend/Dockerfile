# Use an official Node.js runtime as a parent image
FROM node:20-slim AS base

# Set the working directory
WORKDIR /app

# Install pnpm globally
RUN npm install -g pnpm

# Install dependencies stage
FROM base AS deps
WORKDIR /app

# Copy package.json and pnpm-lock.yaml
COPY package.json pnpm-lock.yaml* ./

# Install dependencies using pnpm
RUN pnpm install --frozen-lockfile

# Build stage
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build the Next.js application
RUN pnpm build

# Production stage
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
# Uncomment the following line in case you need sharp support
# RUN apt-get update && apt-get install -y libc6

# Copy the built Next.js application files from the builder stage
COPY --from=builder /app/public ./public
# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Set the user to run the application
# USER nextjs

EXPOSE 3000

ENV PORT=3000
# set hostname to localhost
ENV HOSTNAME="0.0.0.0"

# Command to run the application
CMD ["node", "server.js"] 