"""
Command Line Interface for Crypto Analysis Chatbot.
Provides an interactive terminal-based chat interface.
"""

import sys
import os
import logging
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
from src import CryptoChatbot
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def flush_log_handler(handler):
    """
    Force flush log handler to disk.
    
    Args:
        handler: Logging handler to flush
    """
    if handler:
        handler.flush()
        # Also flush the underlying file stream's OS buffer
        if hasattr(handler, 'stream') and hasattr(handler.stream, 'fileno'):
            try:
                os.fsync(handler.stream.fileno())  # Force OS to write to disk
            except (AttributeError, OSError):
                pass  # Some streams don't support fsync


class CLI:
    """Command-line interface for the chatbot."""
    
    def __init__(self, chatbot: CryptoChatbot, file_handler: Optional[logging.FileHandler] = None):
        """
        Initialize CLI.
        
        Args:
            chatbot: CryptoChatbot instance
            file_handler: File handler to flush after each inference
        """
        self.chatbot = chatbot
        self.console = Console()
        self.running = False
        self.file_handler = file_handler
    
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
                    
                    # Flush logs after each inference to ensure they're written to disk
                    flush_log_handler(self.file_handler)
                
                except KeyboardInterrupt:
                    self.console.print("\n\n[yellow]Interrupted by user[/yellow]")
                    self.running = False
                    break
                
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}", exc_info=True)
                    self.console.print(f"\n[red]Error: {str(e)}[/red]")
                    self.console.print("[yellow]Please try again or type 'exit' to quit.[/yellow]\n")
                    # Flush logs after error
                    flush_log_handler(self.file_handler)
            
            # Print goodbye message
            self.print_goodbye()
            
            # Final flush on exit
            flush_log_handler(self.file_handler)
        
        except Exception as e:
            logger.error(f"Fatal error in CLI: {e}", exc_info=True)
            self.console.print(f"\n[red]Fatal error: {str(e)}[/red]")
            sys.exit(1)


def run_cli(verbose: bool = False, session_id: Optional[str] = None):
    """
    Main entry point for CLI interface.
    
    Args:
        verbose: If True, show INFO logs in console. If False, only show WARNING/ERROR.
    """
    file_handler = setup_logging(verbose=verbose, enable_console=True)
    
    console = Console()
    
    try:
        # Initialize chatbot
        console.print("\n[bold cyan]Initializing chatbot...[/bold cyan]")
        if verbose:
            console.print("[dim]Verbose mode enabled - detailed API logs will be shown[/dim]")
            logger.info("🔊 Verbose logging enabled - You will see all INFO level logs in console")
        else:
            logger.info("🔇 Production mode - Console shows only warnings/errors (all logs saved to chatbot.log)")
        
        default_user_id = os.getenv("CLI_USER_ID", "cli_user")
        chatbot = CryptoChatbot(user_id=default_user_id, session_id=session_id)
        active_session = chatbot.get_session_id()
        console.print(f"[dim]Session ID: {active_session}[/dim]")
        if not session_id:
            logger.info(f"Started new session: {active_session}")
        else:
            logger.info(f"Resumed session: {active_session}")
        
        # Flush logs after initialization
        flush_log_handler(file_handler)
        
        # Create and run CLI with file_handler for efficient flushing
        cli = CLI(chatbot, file_handler)
        cli.run()
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        flush_log_handler(file_handler)  # Ensure logs are written before exit
        sys.exit(0)
    
    except Exception as e:
        console.print(f"\n[red]Failed to start chatbot: {str(e)}[/red]")
        logger.error(f"Failed to start chatbot: {e}", exc_info=True)
        flush_log_handler(file_handler)  # Ensure error logs are written before exit
        sys.exit(1)
        
if __name__ == "__main__":
    run_cli()

