@echo off
echo ========================================
echo RobotAgent MVP 环境配置
echo ========================================
echo.

echo 请按照以下步骤配置您的Qwen API密钥：
echo.
echo 1. 访问阿里云百炼平台：https://bailian.console.aliyun.com/
echo 2. 登录您的阿里云账号
echo 3. 在API管理页面获取您的API Key
echo 4. 将API Key设置为环境变量
echo.

set /p api_key="请输入您的Qwen API Key: "

if "%api_key%"=="" (
    echo 错误：API Key不能为空！
    pause
    exit /b 1
)

echo.
echo 正在设置环境变量...

:: 设置当前会话的环境变量
set DASHSCOPE_API_KEY=%api_key%

:: 设置系统环境变量（需要管理员权限）
setx DASHSCOPE_API_KEY "%api_key%" >nul 2>&1

if %errorlevel% equ 0 (
    echo ✓ 环境变量设置成功！
    echo.
    echo 环境变量 DASHSCOPE_API_KEY 已设置为: %api_key%
    echo.
    echo 注意：
    echo - 当前会话立即生效
    echo - 新的命令行窗口需要重新打开才能使用新的环境变量
    echo - 如果设置失败，请以管理员身份运行此脚本
) else (
    echo ⚠ 系统环境变量设置失败，但当前会话已设置
    echo 请以管理员身份运行此脚本以设置永久环境变量
)

echo.
echo 您现在可以运行 start.bat 启动系统
echo.
pause