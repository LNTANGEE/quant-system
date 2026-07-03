# 量化系统长期稳定部署

你截图里的 `https://quant-system.onrender.com` 返回 `Not Found`，含义很明确：Render 上还没有成功创建并绑定这个 Web Service，或者你打开的不是 Render 实际分配的服务地址。

现在项目已经补齐仓库根目录部署入口：

- 根目录 `Dockerfile`
- 根目录 `render.yaml`
- 根目录 `.dockerignore`
- 根目录 `.gitignore`
- 应用目录 `stock_ai_system/`

这样部署时，Render 会从仓库根目录直接识别服务，不需要手动指定子目录。

## 稳定方案：Render Blueprint

适合手机微信长期打开。

1. 把当前整个目录推送到 GitHub 仓库，不要只上传 `stock_ai_system` 子目录。
2. 打开 Render，选择 `New` -> `Blueprint`。
3. 选择刚才的 GitHub 仓库。
4. Render 会读取根目录的 `render.yaml`。
5. 服务名称为 `quant-system`。
6. 部署完成后，以 Render 页面显示的正式 URL 为准。

如果 `quant-system.onrender.com` 已经被占用或服务没有创建成功，Render 可能会分配类似下面的地址：

```text
https://quant-system-xxxx.onrender.com
```

请以 Render 控制台里的 `Service URL` 为准。

## 当前 Render 配置

```yaml
services:
  - type: web
    name: quant-system
    env: docker
    plan: starter
    autoDeploy: true
    healthCheckPath: /
    disk:
      name: quant-system-data
      mountPath: /app/local_data
      sizeGB: 1
```

SQLite 数据会保存到：

```text
/app/local_data
```

这需要 Render 持久磁盘，建议使用非免费实例，否则自选股、成本价、卖出记录等本地数据可能无法长期保存。

## 为什么临时链接不稳定

`trycloudflare.com` 是临时隧道：

- 电脑关机后会断；
- 网络波动后会断；
- 重启后网址会变；
- 不适合作为长期给微信打开的网址。

长期稳定必须使用 Render、Railway、云服务器或自己的域名。

## 本地运行

```powershell
cd stock_ai_system
pip install -r requirements.txt
streamlit run app.py
```

本地地址：

```text
http://localhost:8501
```

这个地址只能在本机打开，不能作为手机微信长期访问地址。

## 风险提示

本系统只做量化概率分析，不构成投资建议，股市有风险，操作由用户自行承担。
