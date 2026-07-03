# 量化系统长期稳定部署

本系统是 Streamlit 应用，手机微信要直接打开，必须部署到公网地址，不能只用 `localhost`。

## 推荐方案：Render/Railway/Docker 长期部署

项目已包含：

- `Dockerfile`
- `render.yaml`
- `.streamlit/config.toml`
- `Procfile`

当前配置已支持：

- 服务名：`quant-system`
- 云平台动态端口：读取 `$PORT`
- SQLite 数据目录：`/app/local_data`
- Render 持久磁盘挂载：`/app/local_data`

部署步骤：

1. 把 `stock_ai_system` 上传到 GitHub 仓库。
2. 打开 Render 或 Railway，新建 Web Service。
3. 选择 GitHub 仓库，并选择 Docker 部署。
4. 平台会自动读取 `Dockerfile`，端口使用 `$PORT`。
5. 部署完成后，平台会给一个长期可访问的 `https://...` 地址。
6. 把该地址复制到微信即可打开。

Render 如果要保存自选股、成本价和操作记录，建议使用带持久磁盘的实例。当前 `render.yaml` 已配置 `quant-system-data` 磁盘，挂载到 `/app/local_data`。

建议环境变量：

```text
AKSHARE_TIMEOUT_SECONDS=4
FAST_FALLBACK_FIRST=1
TUSHARE_TOKEN=你的Token，可选
```

## 可选方案：Streamlit Community Cloud

1. 上传代码到 GitHub。
2. 打开 https://share.streamlit.io
3. New app。
4. Main file path 填：

```text
app.py
```

5. App URL 生成后，可以直接在微信打开。

注意：免费平台可能会休眠，第一次打开会慢；SQLite 本地文件也可能不适合长期保存关键数据。

## 临时方案：本机 Cloudflare Tunnel

只适合临时给手机微信打开，不适合长期稳定使用。电脑关机、网络断开、进程退出、临时域名过期都会导致链接打不开。

安装 `cloudflared` 后，在项目目录运行：

```powershell
.\run_public_cloudflare.ps1
```

终端出现类似：

```text
https://xxxx.trycloudflare.com
```

复制这个链接到微信即可打开。

长期稳定网址请使用 Render、Railway、云服务器或自己的域名，不要依赖 `trycloudflare.com` 临时链接。

## 数据和安全说明

- 第一版数据库是 SQLite，本地部署时数据保存在本机。
- 云端部署时，免费平台文件系统可能不持久，长期使用建议换成云数据库或挂载持久磁盘。
- 本系统只做量化概率分析，不构成投资建议，股市有风险，操作由用户自行承担。
