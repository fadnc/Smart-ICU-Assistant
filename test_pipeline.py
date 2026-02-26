"""
Quick Test Script for Smart ICU Assistant Pipeline
Tests individual components before running full pipeline
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared loader to avoid re-loading 758K CHARTEVENTS rows between tests
_shared_loader = None
_shared_merged = None


def _get_shared_loader():
    """Get or create a shared MIMICDataLoader instance"""
    global _shared_loader, _shared_merged
    if _shared_loader is None:
        from data_loader import MIMICDataLoader
        _shared_loader = MIMICDataLoader('demo', 'config.yaml')
        _shared_merged = _shared_loader.merge_data()
    return _shared_loader, _shared_merged


def test_data_loader():
    """Test data loading"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Data Loader")
    logger.info("="*60)
    
    t0 = time.time()
    try:
        loader, merged = _get_shared_loader()
        
        logger.info(f"✓ Loaded {len(merged)} ICU stays")
        logger.info(f"✓ {merged['subject_id'].nunique()} unique patients")
        logger.info(f"✓ Chartevents: {len(loader.chartevents)} rows")
        logger.info(f"✓ Labevents: {len(loader.labevents)} rows (with icustay_id)")
        logger.info(f"  Completed in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        logger.error(f"✗ Data loader failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature extraction"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Feature Engineering")
    logger.info("="*60)
    
    t0 = time.time()
    try:
        from feature_engineering import FeatureEngineer
        
        loader, merged = _get_shared_loader()
        fe = FeatureEngineer('config.yaml')
        
        # Test on first stay
        first_stay = merged.iloc[0]
        logger.info(f"  Extracting features for ICU stay {first_stay['icustay_id']}...")
        features = fe.extract_features_for_stay(
            icustay_id=first_stay['icustay_id'],
            icu_intime=first_stay['intime'],
            icu_outtime=first_stay['outtime'],
            chartevents=loader.chartevents,
            labevents=loader.labevents,
            d_items=loader.d_items,
            d_labitems=loader.d_labitems,
            window_hours=6
        )
        
        logger.info(f"✓ Extracted features: shape {features.shape}")
        logger.info(f"  Completed in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        logger.error(f"✗ Feature engineering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test model initialization"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Model Architectures")
    logger.info("="*60)
    
    t0 = time.time()
    try:
        import torch
        import numpy as np
        from models import LSTMModel, TCNModel, XGBoostPredictor
        
        # Test data
        X = torch.randn(16, 24, 30)  # batch, seq_len, features
        
        # Test LSTM
        lstm = LSTMModel(input_size=30, hidden_size=64, num_tasks=6)
        lstm_out = lstm(X)
        logger.info(f"✓ LSTM output shape: {lstm_out.shape}")
        
        # Test TCN
        tcn = TCNModel(input_size=30, num_channels=[32, 64], num_tasks=6)
        tcn_out = tcn(X)
        logger.info(f"✓ TCN output shape: {tcn_out.shape}")
        
        # Test XGBoost
        xgb = XGBoostPredictor(num_tasks=6)
        logger.info(f"✓ XGBoost initialized")
        
        logger.info(f"  Completed in {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        logger.error(f"✗ Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_quick():
    """Quick test of full pipeline with sample"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Quick Pipeline Test (10 ICU stays)")
    logger.info("="*60)
    
    t0 = time.time()
    try:
        from main_pipeline import SmartICUPipeline
        
        pipeline = SmartICUPipeline(config_path='config.yaml', data_dir='demo')
        
        # Run with small sample
        logger.info("Running pipeline with 10 ICU stays...")
        results = pipeline.run(sample_size=10)
        
        if results:
            logger.info(f"✓ Pipeline completed successfully! ({time.time()-t0:.1f}s)")
            return True
        else:
            logger.warning(f"⚠ Pipeline returned no results (might need more data) ({time.time()-t0:.1f}s)")
            return False
            
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("SMART ICU ASSISTANT - COMPONENT TESTS")
    logger.info("="*60)
    
    total_start = time.time()
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Feature Engineering", test_feature_engineering),
        ("Model Architectures", test_models),
        ("Full Pipeline (Quick)", test_full_pipeline_quick)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    logger.info(f"Total time: {time.time()-total_start:.1f}s")
    logger.info("="*60)
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
