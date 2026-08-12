# ── Frontend Dockerfile (React/Vite → Nginx) ───────────────────────────────────

# ---- Aşama 1: Build ----
FROM node:20-alpine AS builder

WORKDIR /app

# Bağımlılıkları yükle
COPY package.json package-lock.json* ./
RUN npm ci

# Kaynak kodunu kopyala ve derle
COPY . .
RUN npm run build

# ---- Aşama 2: Serve (Nginx) ----
FROM nginx:alpine

# Özel Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Build çıktısını Nginx'e kopyala
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]


