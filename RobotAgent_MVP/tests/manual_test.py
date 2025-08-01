import asyncio
import json
import time
import httpx
from typing import Dict, Any

class RobotAgentTester:
    """RobotAgent MVP 测试工具"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"test_session_{int(time.time())}"
        
    async def test_health_check(self) -> bool:
        """测试健康检查"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/health")
                return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    async def test_system_status(self) -> Dict[str, Any]:
        """测试系统状态"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/system-status")
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"状态码: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_command_processing(self, command: str) -> Dict[str, Any]:
        """测试命令处理"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "user_id": "test_user",
                    "input_text": command,
                    "session_id": self.session_id
                }
                
                start_time = time.time()
                response = await client.post(
                    f"{self.base_url}/api/process-command",
                    json=payload,
                    timeout=30.0
                )
                end_time = time.time()
                
                result = {
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "success": response.status_code == 200
                }
                
                if response.status_code == 200:
                    result["data"] = response.json()
                else:
                    result["error"] = response.text
                
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    async def test_memory_records(self) -> Dict[str, Any]:
        """测试记忆记录"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/memory-records")
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"状态码: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """测试性能指标"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/performance-metrics")
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"状态码: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def run_comprehensive_test(self):
        """运行综合测试"""
        print("=== RobotAgent MVP 综合测试 ===\n")
        
        # 1. 健康检查
        print("1. 健康检查测试...")
        health_ok = await self.test_health_check()
        print(f"   结果: {'✓ 通过' if health_ok else '✗ 失败'}\n")
        
        if not health_ok:
            print("服务未启动，请先启动后端服务")
            return
        
        # 2. 系统状态检查
        print("2. 系统状态测试...")
        status = await self.test_system_status()
        if "error" not in status:
            print("   系统状态:")
            for key, value in status.get("data", {}).items():
                print(f"     {key}: {value}")
        else:
            print(f"   错误: {status['error']}")
        print()
        
        # 3. 性能指标检查
        print("3. 性能指标测试...")
        metrics = await self.test_performance_metrics()
        if "error" not in metrics:
            print("   性能指标:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"     {key}: {value}")
        else:
            print(f"   错误: {metrics['error']}")
        print()
        
        # 4. 命令处理测试
        print("4. 命令处理测试...")
        test_commands = [
            "移动机械臂到位置 (0.5, 0.3, 0.2)",
            "抓取桌上的红色方块",
            "将夹爪打开",
            "回到初始位置",
            "停止所有运动"
        ]
        
        for i, command in enumerate(test_commands, 1):
            print(f"   测试命令 {i}: {command}")
            result = await self.test_command_processing(command)
            
            if result["success"]:
                print(f"     ✓ 成功 (响应时间: {result['response_time']:.2f}s)")
                data = result.get("data", {})
                if "data" in data:
                    cmd_data = data["data"]
                    print(f"     解析意图: {cmd_data.get('parsed_intent', 'N/A')}")
                    print(f"     解析动作: {cmd_data.get('parsed_action', 'N/A')}")
            else:
                print(f"     ✗ 失败: {result.get('error', '未知错误')}")
            print()
        
        # 5. 记忆记录检查
        print("5. 记忆记录测试...")
        memory = await self.test_memory_records()
        if "error" not in memory:
            records = memory.get("data", {}).get("records", [])
            print(f"   记忆记录数量: {len(records)}")
            for record in records[:3]:  # 显示前3条
                print(f"     - {record.get('filename', 'N/A')} ({record.get('interaction_count', 0)} 次交互)")
        else:
            print(f"   错误: {memory['error']}")
        print()
        
        print("=== 测试完成 ===")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RobotAgent MVP 测试工具")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--command", help="测试单个命令")
    
    args = parser.parse_args()
    
    tester = RobotAgentTester(args.url)
    
    if args.command:
        # 测试单个命令
        print(f"测试命令: {args.command}")
        result = await tester.test_command_processing(args.command)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # 运行综合测试
        await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())