import numpy as np

def ruku4(function, state, params, dt, steps, noise):
    noise = state.shape[1]*[noise] if state.ndim == 2 else [noise]
    state += np.array([n*np.random.randn(len(n))*np.sqrt(dt) for n in noise]).reshape(state.shape)
    for i in range(steps):
        k1 = dt*function(state,params)
        k2 = dt*function(state+k1/2,params)
        k3 = dt*function(state+k2/2,params)
        k4 = dt*function(state+k3,params)
        state += k1/6. + k2/3. + k3/3. + k4/6.
    return state

def euler_maruyama(function, state, params, dt, steps, noise):
    noise = state.shape[1]*[noise] if state.ndim == 2 else [noise]
    for i in range(steps):
        state += dt*function(state,params) + np.array([n*np.random.randn(len(n))*np.sqrt(dt) for n in noise]).reshape(state.shape)
    return state

def euler(function, state, params, dt, steps, noise):
    noise = state.shape[1]*[noise] if state.ndim == 2 else [noise]
    for i in range(steps):
        state += dt*function(state,params)
    return state

def test_integrator(function, state, params, dt, steps, noise):
    noise = state.shape[1]*[noise] if state.ndim == 2 else [noise]
    for i in range(steps):
        state += np.zeros_like(state)
    print('In test_integrator')
    return state
