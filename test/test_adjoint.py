import numpy as np
import pytest
from src.adjoint import forward_m, adjoint_eq, AdjointFindPDE

def test_forward_m_shape():
    numPDE = 2
    Nx = 5
    Nt = 3
    f0 = np.ones((numPDE, Nx))
    dx = [1.0]
    dt = 0.1
    ps = [(1, 0), (0, 1)]
    ds = [(0, 0)]
    params = np.zeros((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    assert f.shape == (numPDE, Nt, Nx)
    assert isinstance(f, np.ndarray)

def test_forward_m_zero_params_constant_solution():
    numPDE = 1
    Nx = 4
    Nt = 5
    f0 = np.random.rand(numPDE, Nx)
    dx = [1.0]
    dt = 0.1
    ps = [(1,)]
    ds = [(0,)]
    params = np.zeros((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    # Repeat f0 along time axis to match f
    expected = np.repeat(f0[:, None, :], Nt, axis=1)
    np.testing.assert_allclose(f, expected, atol=1e-12)

def test_forward_m_small_dt_changes_little():
    numPDE = 1
    Nx = 4
    Nt = 2
    f0 = np.ones((numPDE, Nx))
    dx = [1.0]
    dt = 1e-8
    ps = [(1,)]
    ds = [(0,)]
    params = np.ones((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    # Change should be very small
    diff = np.abs(f[:,1,:] - f[:,0,:])
    assert np.all(diff < 1e-7)

def test_forward_m_handles_derivatives():
    numPDE = 1
    Nx = 5
    Nt = 2
    f0 = np.arange(Nx).reshape((numPDE, Nx)).astype(float)
    dx = [1.0]
    dt = 0.1
    ps = [(1,)]
    ds = [(1,)]  # first derivative
    params = np.ones((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    # Just check it runs and shape is correct
    assert f.shape == (numPDE, Nt, Nx)

def test_adjoint_eq_terminal_condition():
    numPDE = 2
    Nx = 4
    Nt = 5
    f0 = np.random.rand(numPDE, Nx)
    dx = [1.0]
    dt = 0.1
    ps = [(1,), (2,)]
    ds = [(0,), (1,)]
    params = np.zeros((numPDE, len(ds), len(ps)))

    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    fs = f[:, -1, :]  # target same as final f for simplicity
    lam = adjoint_eq(params, ps, ds, fs, f, dx, dt, Nt)

    # Check shape matches
    assert lam.shape == f.shape
    # Terminal condition: last time step
    np.testing.assert_allclose(lam[:, -1, :], 2*(fs - f[:, -1, :]), atol=1e-12)
    # For zero params, the propagation should keep lam equal to terminal condition backward
    assert np.allclose(lam[:, 0, :], lam[:, -1, :], atol=1e-12)

def test_adjoint_eq_nonzero_params_runs():
    numPDE = 1
    Nx = 3
    Nt = 4
    f0 = np.random.rand(numPDE, Nx)
    dx = [0.5]
    dt = 0.1
    ps = [(1,)]
    ds = [(1,)]
    params = np.random.rand(numPDE, len(ds), len(ps))
    f = forward_m(params, ps, ds, f0, dx, dt, Nt)
    fs = f[:, -1, :] + 0.1  # small difference
    lam = adjoint_eq(params, ps, ds, fs, f, dx, dt, Nt)
    assert lam.shape == f.shape  # still should match

def test_AdjointFindPDE_basic_run():
    numPDE = 1
    Nx = 4
    Nt_fine = 3
    dx = [1.0]
    fs = np.random.rand(numPDE, Nt_fine, Nx)
    x = [np.linspace(0,1,Nx)]
    
    estimated_params, eps, losses = AdjointFindPDE(fs, x, dx, Nt_fine=Nt_fine, epochs=2, beta=0.01)
    
    # Check shapes
    assert estimated_params.shape[0] <= 2  # max epochs
    assert estimated_params.shape[1] == numPDE
    assert estimated_params.shape[2] == 3  # default ds length
    assert estimated_params.shape[3] == 3  # default ps length
    assert len(eps) <= 2
    assert len(losses) <= 2

if __name__ == "__main__":
    pytest.main()

