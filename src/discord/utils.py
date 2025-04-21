# src/discord/utils.py

import discord
from discord import Interaction

# send_discord_request 함수 정의
async def send_discord_request(request_type: str, data: dict) -> bool:
    """
    디스코드 요청을 전송하는 함수입니다.

    Args:
        request_type: 요청의 종류 (예: 'ORDER_CONFIRMATION', 'GENERAL_NOTIFICATION' 등)
        data: 요청에 필요한 데이터 딕셔너리

    Returns:
        bool: 요청 처리 성공 여부
    """
    try:
        # 예시로 채널 ID가 들어있는 data로 요청을 보내는 로직
        channel_id = data.get('channel_id')
        message = data.get('message', 'No message')

        # 디스코드 채널 객체 가져오기
        channel = discord.Client.get_channel(channel_id)
        if not channel:
            return False
        
        # 메시지 전송
        await channel.send(f"Request Type: {request_type} - {message}")
        return True
    except Exception as e:
        # 예외 처리
        print(f"Error in send_discord_request: {e}")
        return False
