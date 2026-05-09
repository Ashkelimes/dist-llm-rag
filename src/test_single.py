import asyncio, aiohttp

async def test():
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            "http://localhost:8080/infer",
            json={"prompt": "Hello", "max_tokens": 20}
        ) as resp:
            print(await resp.json())

asyncio.run(test())