"""
Command Line Interface for Crypto Analysis Chatbot.
Provides an interactive terminal-based chat interface.
"""

import sys
import logging
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from src.chatbot import CryptoChatbot

logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for the chatbot."""
    
    def __init__(self, chatbot: CryptoChatbot):
        """
        Initialize CLI.
        
        Args:
            chatbot: CryptoChatbot instance
        """
        self.chatbot = chatbot
        self.console = Console()
        self.running = False
    
    def print_welcome(self):
        """Print welcome message."""
        welcome_text = """
# 🤖 Crypto Analysis Chatbot

Welcome! I'm your AI-powered cryptocurrency analyst assistant.

## What I Can Help With:
- 📊 **Fundamental Analysis** - Market cap, supply, tokenomics
- 📈 **Price Analysis** - Trends, volatility, support/resistance  
- 📉 **Technical Analysis** - RSI, MACD, moving averages
- 💭 **Sentiment Analysis** - Social metrics, news, Fear & Greed
- ⚖️  **Comparative Analysis** - Compare multiple tokens

## Example Questions:
- "Tell me about Bitcoin"
- "What's Ethereum's price trend?"
- "Compare Bitcoin and Solana"
- "Is the market bullish or bearish?"

Type your question below or 'exit' to quit.
"""
        self.console.print(Markdown(welcome_text))
        self.console.print("─" * 70 + "\n")
    
    def print_goodbye(self):
        """Print goodbye message."""
        goodbye_text = """
## 👋 Thank you for using Crypto Analysis Chatbot!

Stay informed and trade wisely! 🚀
"""
        self.console.print("\n")
        self.console.print(Markdown(goodbye_text))
    
    def format_response(self, response: str) -> str:
        """
        Format the agent's response for better readability.
        
        Args:
            response: Raw response from agent
            
        Returns:
            Formatted response
        """
        # Add some formatting improvements
        response = response.replace("**", "**")  # Keep bold
        response = response.replace("═══", "─" * 60)  # Replace with console-friendly separator
        return response
    
    def run(self):
        """Run the CLI interface."""
        self.running = True
        
        try:
            # Print welcome message
            self.print_welcome()
            
            # Print status
            self.chatbot.print_status()
            
            # Main chat loop
            while self.running:
                try:
                    # Get user input
                    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                    
                    if not user_input.strip():
                        continue
                    
                    # Check for exit commands
                    if user_input.lower().strip() in ["exit", "quit", "bye", "goodbye"]:
                        self.running = False
                        break
                    
                    # Check for special commands
                    if user_input.lower().strip() == "clear":
                        self.chatbot.clear_conversation()
                        self.console.print("[green]✓ Conversation history cleared[/green]\n")
                        continue
                    
                    if user_input.lower().strip() == "status":
                        self.chatbot.print_status()
                        continue
                    
                    if user_input.lower().strip() == "help":
                        self.print_welcome()
                        continue
                    
                    # Show thinking indicator
                    with self.console.status("[bold green]Analyzing...", spinner="dots"):
                        # Get response from chatbot
                        response = self.chatbot.chat(user_input)
                    
                    # Format and display response
                    formatted_response = self.format_response(response)
                    
                    self.console.print("\n[bold green]Assistant:[/bold green]")
                    self.console.print(Panel(
                        Markdown(formatted_response),
                        border_style="green",
                        padding=(1, 2)
                    ))
                
                except KeyboardInterrupt:
                    self.console.print("\n\n[yellow]Interrupted by user[/yellow]")
                    self.running = False
                    break
                
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}", exc_info=True)
                    self.console.print(f"\n[red]Error: {str(e)}[/red]")
                    self.console.print("[yellow]Please try again or type 'exit' to quit.[/yellow]\n")
            
            # Print goodbye message
            self.print_goodbye()
        
        except Exception as e:
            logger.error(f"Fatal error in CLI: {e}", exc_info=True)
            self.console.print(f"\n[red]Fatal error: {str(e)}[/red]")
            sys.exit(1)


def run_cli():
    """
    Main entry point for CLI interface.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('chatbot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    console = Console()
    
    try:
        # Initialize chatbot
        console.print("\n[bold cyan]Initializing chatbot...[/bold cyan]")
        chatbot = CryptoChatbot()
        
        # Create and run CLI
        cli = CLI(chatbot)
        cli.run()
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    
    except Exception as e:
        console.print(f"\n[red]Failed to start chatbot: {str(e)}[/red]")
        logger.error(f"Failed to start chatbot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_cli()

