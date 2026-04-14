"""
C. elegans Locomotion Switching Circuit (SANDBOX C)
Simulates forward/backward/omega turn transitions using Brian2

Based on:
- Kato et al. 2015 (connectome analysis)
- Kaplan et al. 2020 (state transitions)
- Cook et al. 2019 (circuit mechanics)

Circuit includes:
- Command interneurons (AVA, AVB, AVD, AVE, PVC) — bistable switch
- Turn circuit (AIB, RIV, SMB, RIM, AIZ, RIA)
- Motor neurons (DA, VA, DB, VB, DD, VD, SMD, RMD)
- Sensory/rhythm neurons (RIB, DVA, PLM, ALM)

Emergent behaviors: state transitions, bistable switching, conflict resolution
"""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import (
    start_scope, ms, mV, pA, hertz,
    NeuronGroup, Synapses, PoissonGroup,
    StateMonitor, SpikeMonitor, Network,
    defaultclock, run
)

# ============================================================================
# SETUP
# ============================================================================
start_scope()
defaultclock.dt = 0.1 * ms

# Neuron parameters
V_rest = -70 * mV
V_thresh = -55 * mV
V_reset = -70 * mV
refractory = 2 * ms

# Time constants per neuron class (ms)
tau_command = 15  # Command interneurons: fast
tau_inter = 20    # Other interneurons
tau_motor = 10    # Motor neurons: fast
tau_sensory = 30  # Sensory neurons: slow

# ============================================================================
# NEURON GROUPS
# ============================================================================

# Define LIF neuron equations
lif_eqs = '''
dv/dt = (v_rest - v + I_ext + I_syn) / tau : volt
I_ext : amp
I_syn : amp
'''

# Command interneurons (5)
command = NeuronGroup(
    5, lif_eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refractory,
    method='exponential_euler',
    namespace={
        'v_rest': V_rest, 'v_thresh': V_thresh, 'v_reset': V_reset,
        'tau': tau_command * ms
    }
)
command.v = V_rest

# Turn circuit (6)
turn = NeuronGroup(
    6, lif_eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refractory,
    method='exponential_euler',
    namespace={
        'v_rest': V_rest, 'v_thresh': V_thresh, 'v_reset': V_reset,
        'tau': tau_inter * ms
    }
)
turn.v = V_rest

# Motor neurons (8)
motor = NeuronGroup(
    8, lif_eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refractory,
    method='exponential_euler',
    namespace={
        'v_rest': V_rest, 'v_thresh': V_thresh, 'v_reset': V_reset,
        'tau': tau_motor * ms
    }
)
motor.v = V_rest

# Sensory/rhythm neurons (4)
sensory = NeuronGroup(
    4, lif_eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refractory,
    method='exponential_euler',
    namespace={
        'v_rest': V_rest, 'v_thresh': V_thresh, 'v_reset': V_reset,
        'tau': tau_sensory * ms
    }
)
sensory.v = V_rest

# Neuron indices for clarity
# Command: 0=AVA, 1=AVB, 2=AVD, 3=AVE, 4=PVC
# Turn: 0=AIB, 1=RIV, 2=SMB, 3=RIM, 4=AIZ, 5=RIA
# Motor: 0=DA, 1=VA, 2=DB, 3=VB, 4=DD, 5=VD, 6=SMD, 7=RMD
# Sensory: 0=RIB, 1=DVA, 2=PLM, 3=ALM

# ============================================================================
# SYNAPSES (Chemical + Gap Junctions)
# ============================================================================

# Synapse model with weight and type
synapse_eqs_chem = '''
w : 1
is_inhibitory : 1
'''

synapse_eqs_gap = '''
g : siemens
'''

# Helper function to add chemical synapses
def add_chemical_synapse(source, target, src_idx, tgt_idx, weight, is_inhib=False, name=""):
    """Add chemical synapse with STDP learning"""
    syn = Synapses(
        source, target,
        model=synapse_eqs_chem + '''
        dA_pre/dt = -A_pre / tau_stdp : 1
        dA_post/dt = -A_post / tau_stdp : 1
        ''',
        on_pre='''
        A_pre += A_pre_const
        v_post += sign(1 - is_inhibitory*2) * w * mV * (1 + 0.1 * A_post)
        ''',
        on_post='''
        A_post += A_post_const
        w = clip(w + A_pre_const * A_post, 0, 2)
        ''',
        namespace={
            'A_pre_const': 0.01, 'A_post_const': -0.012,
            'tau_stdp': 20 * ms
        },
        method='exponential_euler'
    )
    syn.connect(i=src_idx, j=tgt_idx)
    syn.w = weight
    syn.is_inhibitory = int(is_inhib)
    return syn

# Helper function to add gap junctions
def add_gap_junction(source, target, src_idx, tgt_idx, conductance=0.1*mV):
    """Add electrical synapse (gap junction)"""
    syn = Synapses(
        source, target,
        model=synapse_eqs_gap,
        on_pre='v_post += g * (v_pre - v_post) / mV * mV',
        method='exponential_euler'
    )
    syn.connect(i=src_idx, j=tgt_idx)
    syn.g = conductance
    return syn

synapses = []

# === COMMAND CIRCUIT (Bistable switch: AVA ↔ AVB) ===
synapses.append(add_chemical_synapse(command, command, 0, 1, 2, is_inhib=True, name="AVA→AVB_inhib"))
synapses.append(add_chemical_synapse(command, command, 1, 0, 1, is_inhib=True, name="AVB→AVA_inhib"))

# AVA → Motor (backward motor drive)
synapses.append(add_chemical_synapse(command, motor, 0, 0, 16, name="AVA→DA"))  # DA
synapses.append(add_chemical_synapse(command, motor, 0, 1, 12, name="AVA→VA"))  # VA

# AVB → Motor (forward motor drive)
synapses.append(add_chemical_synapse(command, motor, 1, 2, 11, name="AVB→DB"))  # DB
synapses.append(add_chemical_synapse(command, motor, 1, 3, 14, name="AVB→VB"))  # VB

# === BACKWARD TRIGGERING ===
# AVD, AVE, ALM → AVA
synapses.append(add_chemical_synapse(command, command, 2, 0, 12, name="AVD→AVA"))  # AVD→AVA
synapses.append(add_chemical_synapse(command, command, 3, 0, 5, name="AVE→AVA"))   # AVE→AVA
synapses.append(add_chemical_synapse(sensory, command, 3, 0, 2, name="ALM→AVA"))   # ALM→AVA

# ALM → AVD, AVE
synapses.append(add_chemical_synapse(sensory, command, 3, 2, 6, name="ALM→AVD"))   # ALM→AVD
synapses.append(add_chemical_synapse(sensory, command, 3, 3, 2, name="ALM→AVE"))   # ALM→AVE

# === FORWARD TRIGGERING ===
# PVC, PLM, DVA → AVB
synapses.append(add_chemical_synapse(command, command, 4, 1, 10, name="PVC→AVB"))  # PVC→AVB
synapses.append(add_chemical_synapse(sensory, command, 2, 1, 5, name="PLM→AVB"))   # PLM→AVB
synapses.append(add_chemical_synapse(sensory, command, 2, 4, 7, name="PLM→PVC"))   # PLM→PVC
synapses.append(add_chemical_synapse(sensory, command, 1, 1, 3, name="DVA→AVB"))   # DVA→AVB
synapses.append(add_chemical_synapse(sensory, command, 1, 4, 2, name="DVA→PVC"))   # DVA→PVC

# === TURN CIRCUIT ===
# AIB → AVA, RIM, RIV
synapses.append(add_chemical_synapse(turn, command, 0, 0, 3, name="AIB→AVA"))      # AIB→AVA
synapses.append(add_chemical_synapse(turn, turn, 0, 3, 4, name="AIB→RIM"))        # AIB→RIM
synapses.append(add_chemical_synapse(turn, turn, 0, 1, 5, name="AIB→RIV"))        # AIB→RIV

# RIM → AVA, AVB (inhibitory)
synapses.append(add_chemical_synapse(turn, command, 3, 0, 3, name="RIM→AVA"))      # RIM→AVA
synapses.append(add_chemical_synapse(turn, command, 3, 1, 2, is_inhib=True, name="RIM→AVB"))  # RIM→AVB

# RIV → SMD, RMD
synapses.append(add_chemical_synapse(turn, motor, 1, 6, 3, name="RIV→SMD"))       # RIV→SMD
synapses.append(add_chemical_synapse(turn, motor, 1, 7, 4, name="RIV→RMD"))       # RIV→RMD

# AIZ → AIB, RIA
synapses.append(add_chemical_synapse(turn, turn, 4, 0, 3, name="AIZ→AIB"))        # AIZ→AIB
synapses.append(add_chemical_synapse(turn, turn, 4, 5, 5, name="AIZ→RIA"))        # AIZ→RIA

# RIA → RIV, SMD, RMD
synapses.append(add_chemical_synapse(turn, turn, 5, 1, 3, name="RIA→RIV"))        # RIA→RIV
synapses.append(add_chemical_synapse(turn, motor, 5, 6, 5, name="RIA→SMD"))       # RIA→SMD
synapses.append(add_chemical_synapse(turn, motor, 5, 7, 3, name="RIA→RMD"))       # RIA→RMD

# === MOTOR CROSS-INHIBITION (DA↔DD, VA↔VD, DB↔DD, VB↔VD) ===
# Forward: DB, VB; Backward: DA, VA; Cross: DD, VD
synapses.append(add_chemical_synapse(motor, motor, 0, 4, 4, name="DA→DD"))        # DA→DD
synapses.append(add_chemical_synapse(motor, motor, 1, 5, 3, name="VA→VD"))        # VA→VD
synapses.append(add_chemical_synapse(motor, motor, 2, 4, 2, name="DB→DD"))        # DB→DD
synapses.append(add_chemical_synapse(motor, motor, 3, 5, 5, name="VB→VD"))        # VB→VD

# DD, VD inhibit forward/backward
synapses.append(add_chemical_synapse(motor, motor, 4, 0, 3, is_inhib=True, name="DD→DA"))  # DD→DA
synapses.append(add_chemical_synapse(motor, motor, 4, 1, 2, is_inhib=True, name="DD→VA"))  # DD→VA
synapses.append(add_chemical_synapse(motor, motor, 5, 2, 3, is_inhib=True, name="VD→DB"))  # VD→DB
synapses.append(add_chemical_synapse(motor, motor, 5, 3, 4, is_inhib=True, name="VD→VB"))  # VD→VB

# === RHYTHM FEEDBACK ===
# RIB → AVB, DB
synapses.append(add_chemical_synapse(sensory, command, 0, 1, 2, name="RIB→AVB"))   # RIB→AVB
synapses.append(add_chemical_synapse(sensory, motor, 0, 2, 1, name="RIB→DB"))      # RIB→DB

# SMB → SMD (inhibitory)
synapses.append(add_chemical_synapse(turn, motor, 2, 6, 2, is_inhib=True, name="SMB→SMD"))

# DVA → AVA, AVB (inhibitory)
synapses.append(add_chemical_synapse(sensory, command, 1, 0, 1, is_inhib=True, name="DVA→AVA"))
synapses.append(add_chemical_synapse(sensory, command, 1, 1, 3, name="DVA→AVB"))

# === GAP JUNCTIONS (Electrical synapses) ===
gap_junctions = []

# Command group (AVA-AVA, AVB-AVB, AVA-AVD)
gap_junctions.append(add_gap_junction(command, command, 0, 0, 0.05*mV))  # AVA-AVA
gap_junctions.append(add_gap_junction(command, command, 1, 1, 0.04*mV))  # AVB-AVB
gap_junctions.append(add_gap_junction(command, command, 0, 2, 0.02*mV))  # AVA-AVD

# Motor group (DA-DA, DB-DB, VA-VA, VB-VB)
gap_junctions.append(add_gap_junction(motor, motor, 0, 0, 0.03*mV))      # DA-DA
gap_junctions.append(add_gap_junction(motor, motor, 2, 2, 0.02*mV))      # DB-DB
gap_junctions.append(add_gap_junction(motor, motor, 1, 1, 0.02*mV))      # VA-VA
gap_junctions.append(add_gap_junction(motor, motor, 3, 3, 0.03*mV))      # VB-VB

# ============================================================================
# MONITORS
# ============================================================================

# Monitor all neuron voltages
v_command = StateMonitor(command, 'v', record=True)
v_turn = StateMonitor(turn, 'v', record=True)
v_motor = StateMonitor(motor, 'v', record=True)
v_sensory = StateMonitor(sensory, 'v', record=True)

# Monitor spike times
spike_command = SpikeMonitor(command)
spike_turn = SpikeMonitor(turn)
spike_motor = SpikeMonitor(motor)
spike_sensory = SpikeMonitor(sensory)

# ============================================================================
# NETWORK SETUP
# ============================================================================

net = Network(command, turn, motor, sensory)
net.add(v_command, v_turn, v_motor, v_sensory)
net.add(spike_command, spike_turn, spike_motor, spike_sensory)
net.add(synapses + gap_junctions)

# ============================================================================
# SIMULATION PROTOCOL
# ============================================================================

# 8 phases of stimulation
phases = [
    # (duration_ms, description)
    (500, "Phase 1: FREE RUN (AVB baseline)"),
    (500, "Phase 2: ANTERIOR TOUCH (ALM)"),
    (500, "Phase 3: RELEASE (no stimulus)"),
    (500, "Phase 4: POSTERIOR TOUCH (PLM)"),
    (500, "Phase 5: OMEGA TURN (AIB)"),
    (500, "Phase 6: RECOVERY"),
    (500, "Phase 7: REPEATED ANTERIOR TOUCH"),
    (500, "Phase 8: CONFLICT (ALM + PLM)"),
]

total_time = sum(p[0] for p in phases)

print(f"\n{'='*80}")
print(f"C. elegans Locomotion Switching Circuit Simulation")
print(f"{'='*80}")
print(f"Total simulation time: {total_time} ms")
print(f"Number of neurons: 23 (5 command + 6 turn + 8 motor + 4 sensory)")
print(f"{'='*80}\n")

# Run simulation with phase-specific stimulation
current_time = 0

for phase_idx, (duration, description) in enumerate(phases, 1):
    print(f"Running {description}... ", end='', flush=True)

    # Reset stimulation for all sensory neurons
    sensory.I_ext = 0 * pA

    # Apply phase-specific stimulation
    if phase_idx == 1:  # FREE RUN: AVB baseline (AVB = sensory[0] is wrong, need PVC)
        # Actually: use tonic input to command circuit to establish forward baseline
        command.I_ext[1] = 50 * pA  # AVB tonic input

    elif phase_idx == 2:  # ANTERIOR TOUCH: ALM (sensory[3])
        sensory.I_ext[3] = 200 * pA  # ALM activated

    elif phase_idx == 3:  # RELEASE: no stimulus
        pass  # All I_ext = 0

    elif phase_idx == 4:  # POSTERIOR TOUCH: PLM (sensory[2])
        sensory.I_ext[2] = 200 * pA  # PLM activated

    elif phase_idx == 5:  # OMEGA TURN: AIB (turn[0])
        turn.I_ext[0] = 200 * pA  # AIB directly stimulated

    elif phase_idx == 6:  # RECOVERY: no stimulus
        pass

    elif phase_idx == 7:  # REPEATED ANTERIOR TOUCH: same as phase 2
        sensory.I_ext[3] = 200 * pA  # ALM activated

    elif phase_idx == 8:  # CONFLICT: ALM + PLM simultaneously
        sensory.I_ext[2] = 150 * pA  # PLM
        sensory.I_ext[3] = 150 * pA  # ALM

    # Run this phase
    net.run(duration * ms)
    current_time += duration
    print(f"✓ (t={current_time}ms)")

print(f"\n{'='*80}")
print("Simulation complete!")
print(f"{'='*80}\n")

# ============================================================================
# ANALYSIS & PLOTTING
# ============================================================================

# Extract time arrays (convert to ms)
t_cmd = v_command.t / ms
t_motor = v_motor.t / ms

# Neuron indices for convenience
AVA, AVB, AVD, AVE, PVC = 0, 1, 2, 3, 4
AIB, RIV, SMB, RIM, AIZ, RIA = 0, 1, 2, 3, 4, 5
DA, VA, DB, VB, DD, VD, SMD, RMD = 0, 1, 2, 3, 4, 5, 6, 7
RIB, DVA, PLM, ALM = 0, 1, 2, 3

# ============================================================================
# RASTER PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))

neuron_labels = []
y_offset = 0
colors = {'command': 'red', 'turn': 'blue', 'motor': 'green', 'sensory': 'orange'}

# Command neurons
for i, name in enumerate(['AVA', 'AVB', 'AVD', 'AVE', 'PVC']):
    times = spike_command.t[spike_command.i == i] / ms
    ax.vlines(times, y_offset + 0.3, y_offset + 0.7, color=colors['command'], linewidth=1.5, label='Command' if i == 0 else '')
    neuron_labels.append(name)
    y_offset += 1

# Turn neurons
for i, name in enumerate(['AIB', 'RIV', 'SMB', 'RIM', 'AIZ', 'RIA']):
    times = spike_turn.t[spike_turn.i == i] / ms
    ax.vlines(times, y_offset + 0.3, y_offset + 0.7, color=colors['turn'], linewidth=1.5, label='Turn' if i == 0 else '')
    neuron_labels.append(name)
    y_offset += 1

# Motor neurons
for i, name in enumerate(['DA', 'VA', 'DB', 'VB', 'DD', 'VD', 'SMD', 'RMD']):
    times = spike_motor.t[spike_motor.i == i] / ms
    ax.vlines(times, y_offset + 0.3, y_offset + 0.7, color=colors['motor'], linewidth=1.5, label='Motor' if i == 0 else '')
    neuron_labels.append(name)
    y_offset += 1

# Sensory neurons
for i, name in enumerate(['RIB', 'DVA', 'PLM', 'ALM']):
    times = spike_sensory.t[spike_sensory.i == i] / ms
    ax.vlines(times, y_offset + 0.3, y_offset + 0.7, color=colors['sensory'], linewidth=1.5, label='Sensory' if i == 0 else '')
    neuron_labels.append(name)
    y_offset += 1

ax.set_ylim(0, y_offset)
ax.set_xlim(0, total_time)
ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Neuron', fontsize=12, fontweight='bold')
ax.set_title('C. elegans Locomotion Switching Circuit - Spike Raster', fontsize=14, fontweight='bold')
ax.set_yticks(np.arange(0.5, y_offset, 1))
ax.set_yticklabels(neuron_labels, fontsize=9)
ax.grid(axis='x', alpha=0.3)
ax.legend(loc='upper right')

# Add phase boundaries
phase_times = [0]
for duration, _ in phases[:-1]:
    phase_times.append(phase_times[-1] + duration)

for pt in phase_times[1:]:
    ax.axvline(pt, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('/sessions/gifted-quirky-newton/mnt/Documents/전자두뇌연구소/테스트/샌드박스/sandbox_C_raster.png', dpi=150)
print("Saved raster plot: sandbox_C_raster.png")

# ============================================================================
# VOLTAGE TRACES: KEY NEURONS
# ============================================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 10))

# Command circuit (AVA vs AVB)
ax = axes[0]
ax.plot(t_cmd, v_command.v[AVA] / mV, label='AVA (backward)', linewidth=1.5, color='darkblue')
ax.plot(t_cmd, v_command.v[AVB] / mV, label='AVB (forward)', linewidth=1.5, color='darkred')
ax.axhline(-55, color='gray', linestyle='--', alpha=0.5, label='Threshold')
ax.set_ylabel('Voltage (mV)', fontweight='bold')
ax.set_title('Command Interneurons: AVA ↔ AVB Bistable Switch', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Backward motor (DA, VA)
ax = axes[1]
ax.plot(t_motor, v_motor.v[DA] / mV, label='DA (dorsal backward)', linewidth=1.5, color='darkgreen')
ax.plot(t_motor, v_motor.v[VA] / mV, label='VA (ventral backward)', linewidth=1.5, color='lightgreen')
ax.axhline(-55, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Voltage (mV)', fontweight='bold')
ax.set_title('Backward Motor Neurons (DA, VA)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Forward motor (DB, VB)
ax = axes[2]
ax.plot(t_motor, v_motor.v[DB] / mV, label='DB (dorsal forward)', linewidth=1.5, color='orange')
ax.plot(t_motor, v_motor.v[VB] / mV, label='VB (ventral forward)', linewidth=1.5, color='gold')
ax.axhline(-55, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Voltage (mV)', fontweight='bold')
ax.set_title('Forward Motor Neurons (DB, VB)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Turn circuit (RIV, SMD, RMD)
ax = axes[3]
ax.plot(t_motor, v_motor.v[SMD] / mV, label='SMD (head steering)', linewidth=1.5, color='purple')
ax.plot(t_motor, v_motor.v[RMD] / mV, label='RMD (head muscle)', linewidth=1.5, color='magenta')
ax.axhline(-55, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Voltage (mV)', fontweight='bold')
ax.set_xlabel('Time (ms)', fontweight='bold')
ax.set_title('Turn/Head Motor Neurons (SMD, RMD)', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

# Add phase boundaries to all
for pt in phase_times[1:]:
    for ax in axes:
        ax.axvline(pt, color='gray', linestyle=':', alpha=0.3, linewidth=1)

plt.tight_layout()
plt.savefig('/sessions/gifted-quirky-newton/mnt/Documents/전자두뇌연구소/테스트/샌드박스/sandbox_C_traces.png', dpi=150)
print("Saved voltage traces: sandbox_C_traces.png")

# ============================================================================
# COMPUTE PHASE-BY-PHASE STATISTICS
# ============================================================================

def compute_phase_activity(spike_monitor, t_start_ms, t_end_ms, neuron_idx, name):
    """Count spikes in a time window"""
    spikes = spike_monitor.t[(spike_monitor.i == neuron_idx) &
                             (spike_monitor.t >= t_start_ms * ms) &
                             (spike_monitor.t < t_end_ms * ms)]
    duration_s = (t_end_ms - t_start_ms) / 1000.0
    firing_rate = len(spikes) / duration_s if duration_s > 0 else 0
    return firing_rate

# Compute activity for each phase
phase_results = {}
phase_times_list = [0]
for duration, _ in phases[:-1]:
    phase_times_list.append(phase_times_list[-1] + duration)

for phase_idx, (duration, description) in enumerate(phases):
    t_start = phase_times_list[phase_idx]
    t_end = t_start + duration

    phase_results[phase_idx + 1] = {
        'description': description,
        't_start': t_start,
        't_end': t_end,
        'AVA_rate': compute_phase_activity(spike_command, t_start, t_end, AVA, 'AVA'),
        'AVB_rate': compute_phase_activity(spike_command, t_start, t_end, AVB, 'AVB'),
        'DA_rate': compute_phase_activity(spike_motor, t_start, t_end, DA, 'DA'),
        'VA_rate': compute_phase_activity(spike_motor, t_start, t_end, VA, 'VA'),
        'DB_rate': compute_phase_activity(spike_motor, t_start, t_end, DB, 'DB'),
        'VB_rate': compute_phase_activity(spike_motor, t_start, t_end, VB, 'VB'),
        'RIV_rate': compute_phase_activity(spike_turn, t_start, t_end, RIV, 'RIV'),
        'SMD_rate': compute_phase_activity(spike_motor, t_start, t_end, SMD, 'SMD'),
        'RMD_rate': compute_phase_activity(spike_motor, t_start, t_end, RMD, 'RMD'),
    }

print("\nPhase-by-phase firing rates (Hz):")
print("-" * 100)
for phase_num in sorted(phase_results.keys()):
    data = phase_results[phase_num]
    print(f"Phase {phase_num}: {data['description']}")
    print(f"  AVA={data['AVA_rate']:.2f}, AVB={data['AVB_rate']:.2f}")
    print(f"  DA={data['DA_rate']:.2f}, VA={data['VA_rate']:.2f}, DB={data['DB_rate']:.2f}, VB={data['VB_rate']:.2f}")
    print()

print(f"{'='*80}")
print("All simulation files saved!")
print(f"{'='*80}\n")
