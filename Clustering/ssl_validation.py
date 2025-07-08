from ucimlrepo import fetch_ucirepo
import ssl
import urllib.request
import urllib.parse
from typing import Optional


def disable_ssl_verification():
    try:
        # Create an unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create a custom HTTPS handler with unverified SSL context
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        
        # Install the custom handler globally
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        print("✅ SSL certificate verification disabled globally")
        print("🔒 Warning: This reduces security - use only for trusted sources")
        
    except Exception as e:
        print(f"❌ Failed to disable SSL verification: {e}")
        raise


def enable_ssl_verification():
    try:
        # Create default SSL context (with verification)
        ssl_context = ssl.create_default_context()
        
        # Create default HTTPS handler
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        
        # Install the default handler
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        print("✅ SSL certificate verification re-enabled")
        
    except Exception as e:
        print(f"❌ Failed to re-enable SSL verification: {e}")
        raise


def fetch_with_ssl_bypass(name: Optional[str] = None, id: Optional[int] = None):
    # Store original state
    original_ssl_context = ssl.create_default_context()
    
    try:
        # Temporarily disable SSL verification
        disable_ssl_verification()
        
        # Fetch the dataset
        if name and id:
            raise ValueError('Only specify either dataset name or ID, not both')
        elif name:
            data = fetch_ucirepo(name=name)
        elif id:
            data = fetch_ucirepo(id=id)
        else:
            raise ValueError('Must provide either dataset name or ID')
            
        print(f"✅ Successfully fetched dataset")
        return data
        
    finally:
        # Always re-enable SSL verification for security
        enable_ssl_verification()


def configure_pandas_ssl():
    try:
        # Import pandas to configure its URL handling
        import pandas as pd
        
        # Create unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Monkey patch pandas' URL opener to use unverified SSL
        original_urlopen = urllib.request.urlopen
        
        def patched_urlopen(url, *args, **kwargs):
            if isinstance(url, str) and url.startswith('https://'):
                kwargs['context'] = ssl_context
            return original_urlopen(url, *args, **kwargs)
        
        urllib.request.urlopen = patched_urlopen
        
        print("✅ Pandas SSL configuration applied")
        
    except Exception as e:
        print(f"❌ Failed to configure pandas SSL: {e}")
        raise


# Convenience function for common use case
def setup_uci_ssl_bypass():
    print("🔧 Setting up UCI ML Repository SSL bypass...")
    disable_ssl_verification()
    configure_pandas_ssl()
    print("🎉 Setup complete! UCI ML Repository should now work without SSL issues")
