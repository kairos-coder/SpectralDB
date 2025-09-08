# playwright_conductor.py
"""
The Order of the Phantom Page: Knights who venture into external realms.
The execution arm of the SpectralDb black box.
"""

from typing import Dict, Any, Optional, List
import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import logging
import random
import json
from pathlib import Path
from datetime import datetime
import re

# Set up logging to see the Knights' quests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("PhantomPageKnight")

class PlaywrightConductor:
    """A knight who conducts parleys with external oracles through browser automation."""

    def __init__(self, headless: bool = True, persistent_context_dir: Optional[Path] = None):
        self.headless = headless
        self.persistent_context_dir = persistent_context_dir
        self.playwright = None
        self.browser = None
        self.knight_id = f"Knight-{random.randint(1000, 9999)}"
        self.active_quests = {}

    async def __aenter__(self):
        """Async context manager setup: Arm the Knight."""
        self.playwright = await async_playwright().start()
        # Launch browser with persistent context if requested
        browser_args = {}
        if self.persistent_context_dir:
            self.persistent_context_dir.mkdir(parents=True, exist_ok=True)
            browser_args['user_data_dir'] = str(self.persistent_context_dir)
            
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless, 
            **browser_args
        )
        logger.info(f"⚔️ {self.knight_id} armed and ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager teardown: Knight returns to the castle."""
        await self.browser.close()
        await self.playwright.stop()
        logger.info(f"⚔️ {self.knight_id} mission complete.")

    async def _capture_calamity_screenshot(self, page, oracle_name: str):
        """Capture screenshot for debugging failed quests."""
        screenshots_dir = Path("quest_calamities")
        screenshots_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshots_dir / f"calamity_{oracle_name}_{self.knight_id}_{timestamp}.png"
        try:
            await page.screenshot(path=filename)
            logger.error(f"📸 Calamity screenshot saved: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to capture calamity screenshot: {e}")

    def _sanitize_prophecy(self, raw_text: Optional[str]) -> Optional[str]:
        """Sanitize and normalize the oracle's response."""
        if not raw_text:
            return None
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', raw_text.strip())
        # Basic HTML tag stripping (simple but effective)
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        return cleaned

    async def parley_with_oracle(self, target_config: Dict[str, Any], prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        The core mission: Navigate to a target AI chat, deliver the prompt, and return the prophecy.
        
        Args:
            target_config: A scroll defining the target oracle's realm (URL, UI selectors).
            prompt: The text to deliver to the oracle.
            max_retries: Number of attempts before conceding defeat.
            
        Returns:
            The sanitized text of the oracle's response, or None if the quest failed.
        """
        quest_timeout = target_config.get('timeout_override', 45000) # Default 45s
        extracted_response = None
        attempt = 0
        
        while attempt <= max_retries:
            attempt += 1
            # Create a new, isolated browsing context for each attempt
            context = await self.browser.new_context()
            page = await context.new_page()
            page.set_default_timeout(quest_timeout)
            
            try:
                logger.info(f"⚔️ {self.knight_id} attempt {attempt}/{max_retries+1} for {target_config['name']}...")
                
                # 1. Navigate to the oracle's realm
                await page.goto(target_config['url'])

                # 2. Handle any initial gatekeepers (cookie consents, popups)
                if 'accept_cookies_selector' in target_config:
                    try:
                        await page.click(target_config['accept_cookies_selector'], timeout=5000)
                        logger.info(f"⚔️ {self.knight_id} passed the gatekeepers.")
                    except PlaywrightTimeoutError:
                        logger.warning(f"⚔️ {self.knight_id} found no gatekeepers. Pressing on.")

                # 3. Locate the input scroll and inscribe the prompt
                logger.info(f"⚔️ {self.knight_id} delivering the prompt...")
                
                # Try multiple selector strategies for robustness
                input_selectors = target_config['input_selectors'] if 'input_selectors' in target_config else [target_config['input_selector']]
                input_box = None
                
                for selector in input_selectors:
                    try:
                        input_box = page.locator(selector)
                        await input_box.wait_for(state='visible', timeout=5000)
                        break
                    except Exception:
                        continue
                
                if not input_box:
                    raise Exception(f"Could not find input box with any selector: {input_selectors}")
                
                await input_box.click()
                await input_box.fill('')
                await input_box.type(prompt, delay=30)

                # 4. Send the prompt into the ether
                send_button = page.locator(target_config['submit_selector'])
                await send_button.click()

                # 5. Wait for the oracle's prophecy to appear
                logger.info(f"⚔️ {self.knight_id} awaiting the oracle's response...")
                
                response_selectors = target_config['response_selectors'] if 'response_selectors' in target_config else [target_config['response_selector']]
                response_container = None
                
                for selector in response_selectors:
                    try:
                        response_container = page.locator(selector).last
                        await response_container.wait_for(state="visible", timeout=quest_timeout - 10000)
                        break
                    except Exception:
                        continue
                
                if not response_container:
                    raise Exception(f"Could not find response with any selector: {response_selectors}")

                # 6. Extract the text of the prophecy
                extracted_response = await response_container.text_content()
                extracted_response = self._sanitize_prophecy(extracted_response)
                
                logger.info(f"✅ {self.knight_id} successfully retrieved prophecy from {target_config['name']} (Attempt {attempt})")
                break # Success! Break out of retry loop
                
            except PlaywrightTimeoutError as e:
                logger.error(f"⏰ {self.knight_id}'s quest timed out at the gates of {target_config['name']} (Attempt {attempt}): {e}")
                await self._capture_calamity_screenshot(page, target_config['name'])
            except Exception as e:
                logger.error(f"💥 {self.knight_id}'s quest failed in {target_config['name']} (Attempt {attempt}) due to an unforeseen calamity: {e}")
                await self._capture_calamity_screenshot(page, target_config['name'])
            finally:
                # Close the context, sealing the portal
                await context.close()
            
            if attempt <= max_retries:
                logger.info(f"⚔️ {self.knight_id} preparing for retry...")
                await asyncio.sleep(2 ** attempt) # Exponential backoff: 2s, 4s, etc.
        
        return extracted_response

# Dynamic Registry Loader
def load_oracle_registry(registry_path: str = "oracle_registry.json") -> Dict[str, Any]:
    """Load the grimoire of known oracles from a JSON file."""
    registry_file = Path(registry_path)
    if registry_file.exists():
        try:
            with open(registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load oracle registry: {e}")
            return {}
    return {}

# Enhanced TARGET_REGISTRY with multiple selectors and timeouts
TARGET_REGISTRY = {
    "claude_ai": {
        "name": "Claude.ai",
        "url": "https://claude.ai/chat",
        "input_selectors": [
            "div.ProseMirror",
            "[contenteditable='true']",
            "textarea",
            "input[type='text']"
        ],
        "submit_selector": "button:has-text('Send Message')",
        "response_selectors": [
            "div.contents > div .message-content",
            "[class*='message']",
            "[class*='response']",
            "[class*='content']"
        ],
        "timeout_override": 60000, # Claude can be thoughtful
        "requires_login": True
    },
    # Additional oracles can be added here or loaded dynamically
}

# Example of a Knight's Quest with enhanced logging
async def demonstrate_knightly_quest():
    """A demonstration of a Knight's quest to the Claude.ai oracle."""
    prompt = "What is the fundamental concept behind a system that uses multiple AIs?"
    
    # Use persistent context to maintain login state across quests
    persistent_data_dir = Path("./knight_data")
    
    async with PlaywrightConductor(headless=False, persistent_context_dir=persistent_data_dir) as knight:
        prophecy = await knight.parley_with_oracle(TARGET_REGISTRY['claude_ai'], prompt, max_retries=1)
        
        # Prepare for SpectralDb ingestion
        quest_data = {
            'knight_id': knight.knight_id,
            'oracle_name': TARGET_REGISTRY['claude_ai']['name'],
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'prophecy': prophecy,
            'success': prophecy is not None
        }
        
        print(f"\n🎻 Quest Complete:")
        print(f"Knight: {quest_data['knight_id']}")
        print(f"Oracle: {quest_data['oracle_name']}")
        print(f"Success: {quest_data['success']}")
        if prophecy:
            print(f"\nPrompt: {prompt}")
            print(f"\nProphecy: {prophecy[:200]}...") # First 200 chars
        else:
            print("❌ The oracle remained silent.")

        return quest_data

if __name__ == "__main__":
    # Run the demonstration
    quest_report = asyncio.run(demonstrate_knightly_quest())
