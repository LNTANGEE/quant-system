$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkRoot = Split-Path -Parent $Root
$LogDir = Join-Path $WorkRoot "work"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LocalCloudflared = Join-Path $LogDir "cloudflared.exe"
$Cloudflared = if (Test-Path $LocalCloudflared) {
    Get-Item $LocalCloudflared
} else {
    Get-Command cloudflared -ErrorAction SilentlyContinue
}
if (-not $Cloudflared) {
    Write-Host "cloudflared 未安装。请先安装 Cloudflare Tunnel，或把 cloudflared.exe 放到项目 work 目录或 PATH。"
    Write-Host "下载地址: https://github.com/cloudflare/cloudflared/releases/latest"
    exit 1
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    Write-Host "未找到 python，请先安装 Python 或激活虚拟环境。"
    exit 1
}

$PythonDir = Split-Path -Parent $Python.Source
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[Environment]::SetEnvironmentVariable(
    "Path",
    "$PythonDir;$PythonDir\Scripts;C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0",
    "Process"
)

Start-Process -FilePath $Python.Source `
    -ArgumentList @("-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--server.port", "8501") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $LogDir "streamlit-public.out.log") `
    -RedirectStandardError (Join-Path $LogDir "streamlit-public.err.log")

Start-Sleep -Seconds 5

Write-Host "正在创建临时公网链接，出现 https://*.trycloudflare.com 后即可复制到微信打开。"
& $Cloudflared.FullName tunnel --protocol http2 --url http://127.0.0.1:8501
