"""
Fundamental Analysis Module
Analyzes market cap, supply, volume, tokenomics, and liquidity metrics.
"""

from typing import Dict, Any, Optional
import logging
from src.data_sources.coingecko_client import CoinGeckoClient
from src.data_sources.coinmarketcap_client import CoinMarketCapClient
from src.data_sources.messari_client import MessariClient
from src.utils.formatters import OutputFormatter

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """Performs fundamental analysis on cryptocurrency tokens."""
    
    def __init__(
        self,
        coingecko_client: Optional[CoinGeckoClient] = None,
        coinmarketcap_client: Optional[CoinMarketCapClient] = None,
        messari_client: Optional[MessariClient] = None
    ):
        """
        Initialize Fundamental Analyzer.
        
        Args:
            coingecko_client: CoinGecko API client
            coinmarketcap_client: CoinMarketCap API client
            messari_client: Messari API client
        """
        self.coingecko = coingecko_client
        self.coinmarketcap = coinmarketcap_client
        self.messari = messari_client
        
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis.
        
        Args:
            symbol: Token symbol (e.g., "BTC")
            
        Returns:
            dict: Fundamental analysis results
        """
        try:
            # Gather data from available sources
            cg_data = self._get_coingecko_data(symbol)
            cmc_data = self._get_coinmarketcap_data(symbol)
            messari_data = self._get_messari_data(symbol)
            
            # Merge and analyze data
            merged_data = self._merge_data(cg_data, cmc_data, messari_data)
            
            # Generate analysis
            analysis = {
                "symbol": symbol.upper(),
                "name": merged_data.get("name", "Unknown"),
                "market_metrics": self._analyze_market_metrics(merged_data),
                "supply_metrics": self._analyze_supply_metrics(merged_data),
                "liquidity_metrics": self._analyze_liquidity_metrics(merged_data),
                "valuation_metrics": self._analyze_valuation_metrics(merged_data),
                "summary": self._generate_summary(merged_data),
                "raw_data": merged_data,
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {e}")
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
    
    def _merge_data(
        self, 
        cg_data: Optional[Dict], 
        cmc_data: Optional[Dict],
        messari_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Merge data from multiple sources."""
        merged = {}
        
        # Priority: CoinGecko > CoinMarketCap > Messari
        sources = [cg_data, cmc_data]
        if messari_data:
            sources.append(messari_data.get("metrics", {}))
        
        for source in sources:
            if source:
                for key, value in source.items():
                    if value is not None and key not in merged:
                        merged[key] = value
        
        # Add Messari profile data if available
        if messari_data and "profile" in messari_data:
            merged["profile"] = messari_data["profile"]
        
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