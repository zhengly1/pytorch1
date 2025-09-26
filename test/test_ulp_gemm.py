#!/usr/bin/env python3

import os
import unittest
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

class TestUlpGemm(TestCase):
    
    def setUp(self):
        """Set up test environment"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
    
    def test_ulp_gemm_float32(self):
        """Test ULP GEMM with float32 tensors"""
        # Enable ULP GEMM
        os.environ['ENABLE_CUTLASS_ULP'] = '1'
        
        device = torch.device('cuda:0')
        m, n, k = 32, 32, 32
        
        # Create test matrices
        A = torch.randn(m, k, device=device, dtype=torch.float32)
        B = torch.randn(k, n, device=device, dtype=torch.float32)
        C = torch.randn(m, n, device=device, dtype=torch.float32)
        
        alpha, beta = 2.0, 1.5
        
        # Run with ULP GEMM
        result_ulp = torch.addmm(C, A, B, alpha=alpha, beta=beta)
        
        # Disable ULP GEMM for comparison
        os.environ['ENABLE_CUTLASS_ULP'] = '0'
        
        # Run standard GEMM
        result_std = torch.addmm(C, A, B, alpha=alpha, beta=beta)
        
        # Results should be close but not identical due to ULP perturbation
        self.assertEqual(result_ulp.shape, result_std.shape)
        self.assertFalse(torch.equal(result_ulp, result_std))  # Should be different
        self.assertTrue(torch.allclose(result_ulp, result_std, rtol=1e-5, atol=1e-5))  # But close
        
        # Check that difference is small
        max_diff = torch.max(torch.abs(result_ulp - result_std)).item()
        self.assertLess(max_diff, 1e-4)  # Difference should be small
    
    def test_ulp_gemm_half(self):
        """Test ULP GEMM with half precision tensors"""
        # Enable ULP GEMM
        os.environ['ENABLE_CUTLASS_ULP'] = '1'
        
        device = torch.device('cuda:0')
        m, n, k = 32, 32, 32
        
        # Create test matrices
        A = torch.randn(m, k, device=device, dtype=torch.float16)
        B = torch.randn(k, n, device=device, dtype=torch.float16)
        C = torch.randn(m, n, device=device, dtype=torch.float16)
        
        alpha, beta = 2.0, 1.5
        
        # Run with ULP GEMM
        result_ulp = torch.addmm(C, A, B, alpha=alpha, beta=beta)
        
        # Disable ULP GEMM for comparison
        os.environ['ENABLE_CUTLASS_ULP'] = '0'
        
        # Run standard GEMM
        result_std = torch.addmm(C, A, B, alpha=alpha, beta=beta)
        
        # Results should be close but not identical due to ULP perturbation
        self.assertEqual(result_ulp.shape, result_std.shape)
        self.assertFalse(torch.equal(result_ulp, result_std))  # Should be different
        self.assertTrue(torch.allclose(result_ulp, result_std, rtol=1e-3, atol=1e-3))  # But close
        
        # Check that difference is small (larger tolerance for half precision)
        max_diff = torch.max(torch.abs(result_ulp - result_std)).item()
        self.assertLess(max_diff, 1e-2)  # Difference should be small
    
    def test_ulp_gemm_fallback(self):
        """Test that ULP GEMM falls back gracefully when needed"""
        # Enable ULP GEMM
        os.environ['ENABLE_CUTLASS_ULP'] = '1'
        
        device = torch.device('cuda:0')
        
        # Test with various matrix sizes including edge cases
        test_sizes = [(1, 1, 1), (16, 16, 16), (64, 64, 64)]
        
        for m, n, k in test_sizes:
            with self.subTest(m=m, n=n, k=k):
                A = torch.randn(m, k, device=device, dtype=torch.float32)
                B = torch.randn(k, n, device=device, dtype=torch.float32)
                C = torch.randn(m, n, device=device, dtype=torch.float32)
                
                # This should work regardless of ULP GEMM success/failure
                result = torch.addmm(C, A, B)
                self.assertEqual(result.shape, (m, n))
                self.assertTrue(torch.isfinite(result).all())
    
    def test_ulp_gemm_environment_control(self):
        """Test that environment variable properly controls ULP GEMM"""
        device = torch.device('cuda:0')
        m, n, k = 32, 32, 32
        
        A = torch.randn(m, k, device=device, dtype=torch.float32)
        B = torch.randn(k, n, device=device, dtype=torch.float32)
        C = torch.randn(m, n, device=device, dtype=torch.float32)
        
        # Test with ULP disabled (default)
        os.environ['ENABLE_CUTLASS_ULP'] = '0'
        result_disabled = torch.addmm(C, A, B)
        
        # Test with ULP enabled
        os.environ['ENABLE_CUTLASS_ULP'] = '1'
        result_enabled = torch.addmm(C, A, B)
        
        # Results should be different when ULP is enabled
        self.assertFalse(torch.equal(result_disabled, result_enabled))
        self.assertTrue(torch.allclose(result_disabled, result_enabled, rtol=1e-5, atol=1e-5))
    
    def test_ulp_gemm_transpose_variants(self):
        """Test ULP GEMM with transposed matrices"""
        os.environ['ENABLE_CUTLASS_ULP'] = '1'
        
        device = torch.device('cuda:0')
        m, n, k = 32, 32, 32
        
        # Test different transpose combinations
        A = torch.randn(m, k, device=device, dtype=torch.float32)
        B = torch.randn(k, n, device=device, dtype=torch.float32)
        C = torch.randn(m, n, device=device, dtype=torch.float32)
        
        # Standard: no transpose
        result1 = torch.addmm(C, A, B)
        
        # Transpose A
        result2 = torch.addmm(C, A.t(), B)
        
        # Transpose B
        result3 = torch.addmm(C, A, B.t())
        
        # All should complete successfully
        self.assertTrue(torch.isfinite(result1).all())
        self.assertTrue(torch.isfinite(result2).all())
        self.assertTrue(torch.isfinite(result3).all())
    
    def tearDown(self):
        """Clean up test environment"""
        # Reset environment variable
        os.environ['ENABLE_CUTLASS_ULP'] = '0'

if __name__ == "__main__":
    run_tests()