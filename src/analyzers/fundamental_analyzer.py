"""
Fundamental Analysis Module
Analyzes market cap, supply, volume, tokenomics, and liquidity metrics.
"""

from typing import Dict, Any, Optional
import logging
from src.data_sources.coingecko_client import CoinGeckoClient
from src.data_sources.coinmarketcap_client import CoinMarketCapClient
from src.data_sources.messari_client import MessariClient
from src.data_sources.cryptopanic_client import CryptoPanicClient
from src.data_sources.fear_greed_client import FearGreedClient
from src.utils.formatters import OutputFormatter

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """Performs fundamental analysis on cryptocurrency tokens."""
    
    def __init__(
        self,
        coingecko_client: Optional[CoinGeckoClient] = None,
        coinmarketcap_client: Optional[CoinMarketCapClient] = None,
        messari_client: Optional[MessariClient] = None,
        cryptopanic_client: Optional[CryptoPanicClient] = None,
        alternative_client: Optional[FearGreedClient] = None
    ):
        """
        Initialize Fundamental Analyzer.
        
        Args:
            coingecko_client: CoinGecko API client
            coinmarketcap_client: CoinMarketCap API client
            messari_client: Messari API client
            cryptopanic_client: CryptoPanic API client (for major news events)
            alternative_client: Alternative.me client (for global market context)
        """
        self.coingecko = coingecko_client
        self.coinmarketcap = coinmarketcap_client
        self.messari = messari_client
        self.cryptopanic = cryptopanic_client
        self.alternative = alternative_client
        
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Fundamental analysis results
        """
        try:
            logger.info(f"🔍 [FundamentalAnalyzer] Starting analysis for {symbol}")
            
            # Gather data from available sources
            cg_data = self._get_coingecko_data(symbol)
            cmc_data = self._get_coinmarketcap_data(symbol)
            messari_data = self._get_messari_data(symbol)
            
            # Log data source results
            sources_used = []
            if cg_data: sources_used.append("CoinGecko")
            if cmc_data: sources_used.append("CoinMarketCap")
            if messari_data: sources_used.append("Messari")
            if self.alternative: sources_used.append("Alternative.me (global context)")
            if self.cryptopanic: sources_used.append("CryptoPanic (news)")
            logger.info(f"📊 [FundamentalAnalyzer] Data sources used: {', '.join(sources_used) if sources_used else 'None'}")
            
            # Merge and analyze data
            merged_data = self._merge_data(cg_data, cmc_data, messari_data)
            
            # Get important news that might affect fundamentals
            important_news = self._get_important_news(symbol)
            
            # Get global market context for perspective
            market_context = self._get_global_market_context()
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "name": merged_data.get("name", "Unknown"),
                "market_metrics": self._analyze_market_metrics(merged_data),
                "supply_metrics": self._analyze_supply_metrics(merged_data),
                "liquidity_metrics": self._analyze_liquidity_metrics(merged_data),
                "valuation_metrics": self._analyze_valuation_metrics(merged_data),
                "market_context": market_context,
                "important_news": important_news,
                "summary": self._generate_summary(merged_data),
                "raw_data": merged_data,
            }
            
            logger.info(f"✅ [FundamentalAnalyzer] Analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ [FundamentalAnalyzer] Analysis failed for {symbol}: {e}")
            raise
    
    def _get_coingecko_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from CoinGecko."""
        if not self.coingecko:
            return None
        
        try:
            return self.coingecko.get_token_data(symbol)
        except Exception as e:
            logger.warning(f"CoinGecko data fetch failed: {e}")
            return None
    
    def _get_coinmarketcap_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from CoinMarketCap."""
        if not self.coinmarketcap:
            return None
        
        try:
            return self.coinmarketcap.get_token_data(symbol)
        except Exception as e:
            logger.warning(f"CoinMarketCap data fetch failed: {e}")
            return None
    
    def _get_messari_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Messari."""
        if not self.messari:
            return None
        
        try:
            return self.messari.get_token_data(symbol)
        except Exception as e:
            logger.warning(f"Messari data fetch failed: {e}")
            return None
    
    def _get_important_news(self, symbol: str) -> Dict[str, Any]:
        """
        Get important news that might affect fundamental value.
        
        Args:
            symbol: Token symbol
            
        Returns:
            dict: Important news with high panic scores
        """
        if not self.cryptopanic:
            return {"available": False, "note": "News data not available"}
        
        try:
            # Get important/high-impact news
            important_news = self.cryptopanic.get_important_news(currencies=symbol)
            
            if not important_news:
                return {"available": True, "count": 0, "articles": []}
            
            # Filter for high panic score articles (fundamental impact)
            high_impact_news = []
            for article in important_news[:5]:  # Top 5
                panic_score = article.get("panic_score")
                if panic_score and panic_score > 40:  # Moderate+ impact
                    high_impact_news.append({
                        "title": article.get("title"),
                        "published_at": article.get("published_at"),
                        "panic_score": panic_score,
                        "url": article.get("url"),
                        "source": article.get("source", {}).get("title"),
                        "votes": article.get("votes", {}),
                    })
            
            return {
                "available": True,
                "count": len(high_impact_news),
                "articles": high_impact_news,
                "interpretation": self._interpret_news_impact(high_impact_news),
            }
            
        except Exception as e:
            logger.warning(f"Important news fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _interpret_news_impact(self, news_articles: list) -> str:
        """Interpret the impact of recent important news on fundamentals."""
        if not news_articles:
            return "No significant fundamental news events recently"
        
        avg_panic = sum(a.get("panic_score", 0) for a in news_articles) / len(news_articles)
        
        if avg_panic > 70:
            return "Major fundamental news events - Potential significant impact on valuation"
        elif avg_panic > 50:
            return "Notable fundamental developments - Moderate impact expected"
        else:
            return "Some important news - Minor to moderate fundamental impact"
    
    def _get_global_market_context(self) -> Dict[str, Any]:
        """
        Get global market context from Alternative.me.
        
        Provides context for understanding token's fundamentals relative to market:
        - Total crypto market cap (overall market health)
        - BTC dominance (risk sentiment)
        - Active cryptocurrencies (competition)
        """
        if not self.alternative:
            return {"available": False, "note": "Global market context not available"}
        
        try:
            global_data = self.alternative.get_global_data()
            
            btc_dominance = global_data.get("bitcoin_dominance")
            total_market_cap = global_data.get("total_market_cap")
            
            # Interpret market context for fundamentals
            interpretation = self._interpret_market_context_fundamentals(btc_dominance)
            
            return {
                "available": True,
                "total_market_cap": total_market_cap,
                "total_market_cap_formatted": OutputFormatter.format_large_number(total_market_cap) if total_market_cap else "N/A",
                "bitcoin_dominance": btc_dominance,
                "active_cryptocurrencies": global_data.get("active_cryptocurrencies"),
                "active_markets": global_data.get("active_markets"),
                "interpretation": interpretation,
            }
            
        except Exception as e:
            logger.warning(f"Global market context fetch failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _interpret_market_context_fundamentals(self, btc_dominance: Optional[float]) -> str:
        """Interpret market context from fundamental perspective."""
        if not btc_dominance:
            return "Unable to assess market context"
        
        if btc_dominance > 60:
            return "High BTC dominance - Flight to quality, challenging environment for altcoins"
        elif btc_dominance > 50:
            return "Moderate-high BTC dominance - Conservative market favoring established projects"
        elif btc_dominance > 40:
            return "Balanced BTC dominance - Healthy market for quality altcoin fundamentals"
        else:
            return "Low BTC dominance - Risk-on environment, favorable for strong altcoin fundamentals"
    
    def _merge_data(
        self, 
        cg_data: Optional[Dict], 
        cmc_data: Optional[Dict],
        messari_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Merge data from multiple sources.
        
        ⚠️ Messari free tier only provides asset details (no metrics).
        """
        merged = {}
        
        # Priority: CoinGecko > CoinMarketCap
        sources = [cg_data, cmc_data]
        
        for source in sources:
            if source:
                for key, value in source.items():
                    if value is not None and key not in merged:
                        merged[key] = value
        
        # Add Messari details if available (free tier: details only)
        if messari_data and "details" in messari_data:
            details = messari_data["details"]
            # Merge in Messari-specific fields
            merged["messari_category"] = details.get("category")
            merged["messari_sector"] = details.get("sector")
            merged["messari_tagline"] = details.get("tagline")
            merged["messari_description"] = details.get("description")
            merged["messari_tags"] = details.get("tags", [])
            merged["messari_technology"] = details.get("technology")
        
        return merged
    
    def _analyze_market_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market-related metrics."""
        market_cap = data.get("market_cap")
        rank = data.get("market_cap_rank")
        price = data.get("price_usd")
        volume_24h = data.get("total_volume")
        
        # Calculate volume/market cap ratio
        volume_mc_ratio = None
        if market_cap and volume_24h and market_cap > 0:
            volume_mc_ratio = (volume_24h / market_cap) * 100
        
        # Determine market cap category
        market_cap_category = self._categorize_market_cap(market_cap)
        
        return {
            "price": price,
            "market_cap": market_cap,
            "market_cap_formatted": OutputFormatter.format_large_number(market_cap) if market_cap else "N/A",
            "market_cap_rank": rank,
            "market_cap_category": market_cap_category,
            "volume_24h": volume_24h,
            "volume_24h_formatted": OutputFormatter.format_large_number(volume_24h) if volume_24h else "N/A",
            "volume_to_market_cap_ratio": volume_mc_ratio,
            "dominance": data.get("market_cap_dominance"),
        }
    
    def _analyze_supply_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supply-related metrics."""
        circulating = data.get("circulating_supply")
        total = data.get("total_supply")
        max_supply = data.get("max_supply")
        
        # Calculate supply percentages
        circulating_pct = None
        if circulating and max_supply and max_supply > 0:
            circulating_pct = (circulating / max_supply) * 100
        
        # Determine inflation status
        is_inflationary = max_supply is None or (total and max_supply and total < max_supply)
        
        return {
            "circulating_supply": circulating,
            "total_supply": total,
            "max_supply": max_supply,
            "circulating_percentage": circulating_pct,
            "is_inflationary": is_inflationary,
            "supply_model": "Capped" if max_supply else "Uncapped/Inflationary",
            "annual_inflation": data.get("annual_inflation"),
        }
    
    def _analyze_liquidity_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity-related metrics."""
        volume_24h = data.get("total_volume")
        market_cap = data.get("market_cap")
        
        # Volume to market cap ratio (liquidity indicator)
        liquidity_ratio = None
        liquidity_rating = "Unknown"
        
        if volume_24h and market_cap and market_cap > 0:
            liquidity_ratio = (volume_24h / market_cap) * 100
            
            # Rate liquidity
            if liquidity_ratio > 10:
                liquidity_rating = "Excellent"
            elif liquidity_ratio > 5:
                liquidity_rating = "Good"
            elif liquidity_ratio > 2:
                liquidity_rating = "Fair"
            else:
                liquidity_rating = "Low"
        
        return {
            "volume_24h": volume_24h,
            "liquidity_ratio": liquidity_ratio,
            "liquidity_rating": liquidity_rating,
            "interpretation": self._interpret_liquidity(liquidity_ratio),
        }
    
    def _analyze_valuation_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation-related metrics."""
        market_cap = data.get("market_cap")
        fdv = data.get("fully_diluted_valuation")
        price = data.get("price_usd")
        
        # FDV to Market Cap ratio
        fdv_mc_ratio = None
        if fdv and market_cap and market_cap > 0:
            fdv_mc_ratio = fdv / market_cap
        
        return {
            "fully_diluted_valuation": fdv,
            "fdv_formatted": OutputFormatter.format_large_number(fdv) if fdv else "N/A",
            "fdv_to_mc_ratio": fdv_mc_ratio,
            "valuation_assessment": self._assess_valuation(fdv_mc_ratio),
            "ath": data.get("ath"),
            "atl": data.get("atl"),
        }
    
    def _categorize_market_cap(self, market_cap: Optional[float]) -> str:
        """Categorize token by market cap size."""
        if not market_cap:
            return "Unknown"
        
        if market_cap >= 10_000_000_000:  # $10B+
            return "Large Cap"
        elif market_cap >= 1_000_000_000:  # $1B - $10B
            return "Mid Cap"
        elif market_cap >= 100_000_000:  # $100M - $1B
            return "Small Cap"
        else:
            return "Micro Cap"
    
    def _interpret_liquidity(self, ratio: Optional[float]) -> str:
        """Interpret liquidity ratio."""
        if not ratio:
            return "Unable to assess liquidity"
        
        if ratio > 10:
            return "Very high liquidity - easy to buy/sell large amounts"
        elif ratio > 5:
            return "Good liquidity - healthy trading activity"
        elif ratio > 2:
            return "Moderate liquidity - sufficient for most trading"
        else:
            return "Low liquidity - may have difficulty with large trades"
    
    def _assess_valuation(self, fdv_mc_ratio: Optional[float]) -> str:
        """Assess valuation based on FDV/MC ratio."""
        if not fdv_mc_ratio:
            return "Unable to assess valuation"
        
        if fdv_mc_ratio > 3:
            return "High unlock risk - significant tokens yet to enter circulation"
        elif fdv_mc_ratio > 1.5:
            return "Moderate unlock risk - some tokens yet to be released"
        elif fdv_mc_ratio > 1.1:
            return "Low unlock risk - most tokens in circulation"
        else:
            return "Minimal unlock risk - nearly all tokens circulating"
    
    def _generate_summary(self, data: Dict[str, Any]) -> str:
        """Generate a comprehensive summary of fundamental analysis."""
        name = data.get("name", "Unknown")
        symbol = data.get("symbol", "").upper()
        
        market_cap = data.get("market_cap")
        rank = data.get("market_cap_rank")
        price = data.get("price_usd")
        
        summary_parts = []
        
        # Basic info
        summary_parts.append(
            f"{name} ({symbol}) is ranked #{rank} by market capitalization" if rank 
            else f"{name} ({symbol})"
        )
        
        # Market cap
        if market_cap:
            mc_formatted = OutputFormatter.format_large_number(market_cap)
            category = self._categorize_market_cap(market_cap)
            summary_parts.append(f"with a market cap of {mc_formatted} ({category})")
        
        # Supply
        circ_supply = data.get("circulating_supply")
        max_supply = data.get("max_supply")
        
        if circ_supply and max_supply:
            pct = (circ_supply / max_supply) * 100
            summary_parts.append(
                f"{pct:.1f}% of max supply ({OutputFormatter.format_large_number(circ_supply)} / "
                f"{OutputFormatter.format_large_number(max_supply)}) is currently circulating"
            )
        elif circ_supply:
            summary_parts.append(f"Circulating supply: {OutputFormatter.format_large_number(circ_supply)}")
        
        return ". ".join(summary_parts) + "."