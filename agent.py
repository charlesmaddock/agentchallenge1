import asyncio
from dendrite_sdk import DendriteBrowser
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from loguru import logger


# summerize https://www.bbc.com/news/articles/c5ypr3vd7x9o as an exciting x post and upload it to google docs


async def get_and_summerize_news_article(url: str) -> str:
    """Get and summarize a news article from a given URL."""
    async with DendriteBrowser() as dendrite_browser:
        logger.enable("dendrite_sdk")

        page = await dendrite_browser.goto(url)
        is_news = await page.ask("Does this page contain a news article?", bool)
        if not is_news:
            raise ValueError("Page is not a news article")

        content = await page.extract("Extract the content of the news article", str)

        llm = OpenAI(model="gpt-4o")

        user_msg = (
            "Summarize this news article, focusing on the most interesting, "
            "surprising, or controversial aspects. Highlight any shocking "
            "statistics or unexpected twists. The summary should be engaging "
            "and suitable for a social media post on X (Twitter). "
            f"Content: {content}"
        )

        summary = await llm.acomplete(user_msg)  # Using acomplete instead of chat
        print(summary)
        return str(summary)


async def upload_post_to_google_docs(content: str) -> str:
    """Upload a post to Google Docs. If authentication is required, ask the user to authenticate."""
    async with DendriteBrowser() as dendrite_browser:
        logger.enable("dendrite_sdk")
        page = await dendrite_browser.goto(
            "https://docs.google.com/document/d/19xSxTQWkSiP7vtKDhOjenrZJdz8escsVyZwGJrXDXes/edit?usp=sharing",
            expected_page="Should be a Google Docs document, not the login page. If this is the login page, ask the user to authenticate.",
        )
        await page.wait_for("The Google Docs page should be loaded and ready to edit.")
        el = await page.get_element("The editable area of the Google Docs page")
        await el.click()
        await page.keyboard.type(content)
        await asyncio.sleep(4)

    return "Content successfully posted to Google Docs."


get_news_tool = FunctionTool.from_defaults(async_fn=get_and_summerize_news_article)
upload_tool = FunctionTool.from_defaults(async_fn=upload_post_to_google_docs)


llm = OpenAI(model="gpt-4o")
agent = ReActAgent.from_tools([get_news_tool, upload_tool], llm=llm, verbose=True)


async def run_agent():
    while True:
        message = input("Enter a message (or 'quit' to exit): ")
        if message.lower() == "quit":
            break

        response = await agent.achat(message)
        print("Agent response:", response.response)
        print()


if __name__ == "__main__":
    asyncio.run(run_agent())
