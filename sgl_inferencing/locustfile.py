"""
Locust Load Test for SGLang Server
Model: Qwen/Qwen3-4B-Instruct-2507
Target: 10 users, 10 minutes
Prompts: 100 Irodov-style hard physics & math problems
"""

import random
from locust import HttpUser, task, between

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

IRODOV_PROMPTS = [
    # Mechanics
    "A particle moves along the x-axis with velocity v(t) = a - bt where a=6 m/s and b=2 m/s². Find the distance covered in t=4s and the displacement.",
    "Two bodies of masses m1=2kg and m2=3kg are connected by a string over a frictionless pulley. Find the acceleration of the system and tension in the string.",
    "A ball is thrown horizontally from a height h=20m with initial velocity v0=15 m/s. Find the time of flight, range, and velocity at impact.",
    "A car travels in a circle of radius R=100m. If the speed increases at rate a_t=1 m/s² and speed at an instant is v=20 m/s, find total acceleration.",
    "A block of mass m=5kg rests on an incline of angle 30°. Coefficient of static friction μ=0.4. Find the force required to push it up the incline.",
    "A rocket ejects mass at rate dm/dt=500 kg/s with exhaust velocity u=3000 m/s. If initial mass is 10000 kg, find initial thrust and acceleration.",
    "A simple pendulum of length L=1m is displaced 10° from equilibrium. Find period, max velocity at bottom, and max acceleration.",
    "Two particles collide head-on elastically. Mass m1=1kg moving at 4 m/s hits stationary m2=2kg. Find velocities after collision.",
    "A disc of moment of inertia I=2 kg·m² rotates at ω=10 rad/s. A torque of τ=4 N·m is applied. Find angular acceleration and time to stop.",
    "A spring of constant k=500 N/m is compressed by x=0.1m and releases a mass m=0.5kg. Find the velocity of mass when spring returns to natural length.",

    # Thermodynamics
    "One mole of ideal gas expands isothermally at T=300K from V1=1L to V2=10L. Calculate work done by the gas and heat absorbed.",
    "An ideal gas undergoes adiabatic expansion. Initial state: P1=10 atm, T1=500K, V1=2L. Find final temperature if V2=20L. γ=1.4.",
    "A Carnot engine operates between T_H=600K and T_C=300K. If it absorbs Q_H=1000J per cycle, find work output and efficiency.",
    "Calculate the RMS speed of nitrogen molecules at T=300K. Molar mass of N2 = 28 g/mol, R = 8.314 J/mol·K.",
    "One mole of van der Waals gas with a=0.364 J·m³/mol², b=4.27×10⁻⁵ m³/mol expands from V1=1L to V2=10L isothermally at 300K. Find work done.",
    "A heat pump operates between 0°C and 25°C. Find its COP and the electrical energy needed to deliver 1000J of heat to the warm reservoir.",
    "Calculate the entropy change when 1 mole of water is heated from 20°C to 100°C and then vaporized. Cp(water)=75.3 J/mol·K, ΔH_vap=40.7 kJ/mol.",
    "An ideal gas undergoes a cyclic process: isothermal expansion at T1, isochoric cooling to T2, isobaric compression back to start. Find net work done.",
    "Two identical blocks at T1=400K and T2=200K are brought into thermal contact. Find final temperature and total entropy change.",
    "Using Maxwell speed distribution, derive the most probable speed, mean speed, and RMS speed for ideal gas molecules at temperature T.",

    # Electrostatics
    "Two point charges q1=+3μC and q2=-5μC are placed 0.2m apart. Find the electric field at the midpoint and the force between them.",
    "A uniformly charged sphere of radius R=0.1m has total charge Q=10μC. Find electric field inside (at r=0.05m) and outside (at r=0.2m).",
    "A parallel plate capacitor has plates of area A=0.01m², separation d=1mm, and dielectric constant κ=5. Find capacitance and energy stored at V=100V.",
    "Find the potential at the center of a uniformly charged ring of radius R=0.5m with total charge Q=2μC.",
    "An electric dipole of moment p=5×10⁻³⁰ C·m is placed in a uniform field E=10⁶ V/m at 30° to the field. Find torque and potential energy.",
    "Three charges +q, -2q, +q are placed at corners of an equilateral triangle of side a. Find the electric potential and field at the centroid.",
    "A spherical conductor of radius r1=0.05m is concentric with a spherical shell of radius r2=0.1m. Find capacitance of this spherical capacitor.",
    "Calculate the work done to assemble four equal charges q=1μC at the corners of a square of side a=1m.",
    "A long cylinder of radius R=2cm has uniform volume charge density ρ=10⁻⁶ C/m³. Using Gauss's law, find E inside and outside.",
    "Two capacitors C1=4μF and C2=6μF are connected in series across 100V. Find charge, voltage, and energy stored in each.",

    # Magnetism
    "A long straight wire carries current I=10A. Find the magnetic field at distance r=5cm from the wire and force per unit length on a parallel wire carrying 5A at the same distance.",
    "A circular loop of radius R=0.1m carries current I=5A. Find the magnetic field at the center and on the axis at distance x=0.1m.",
    "An electron moves with v=10⁶ m/s perpendicular to a magnetic field B=0.1T. Find the radius of circular motion and cyclotron frequency.",
    "A solenoid of length L=0.5m, radius r=2cm, and n=1000 turns/m carries I=2A. Find the magnetic field inside and the inductance.",
    "A rectangular loop of dimensions a=0.1m × b=0.2m carrying current I=3A is placed in a uniform field B=0.5T. Find max torque and orientation for zero torque.",
    "Find the mutual inductance between two coaxial solenoids of same length L=0.3m, with n1=500 turns/m, n2=800 turns/m, and radius r=3cm.",
    "A conducting rod of length L=0.5m moves with v=2 m/s perpendicular to a field B=0.3T. Find the EMF induced and current if resistance R=5Ω.",
    "An LCR series circuit has L=0.1H, C=100μF, R=10Ω. Find resonant frequency, Q-factor, and impedance at resonance.",
    "Using Ampere's law, find the magnetic field inside a toroid with N=500 turns, mean radius r=0.2m, carrying current I=4A.",
    "A charged particle q=1.6×10⁻¹⁹C, m=1.67×10⁻²⁷kg enters perpendicular to B=1T field. Find pitch of helical path if v has components v_perp=10⁶ and v_parallel=5×10⁵ m/s.",

    # Optics
    "A convex lens of focal length f=20cm forms an image of an object placed 30cm from the lens. Find image distance, magnification, and nature of image.",
    "Light travels from glass (n=1.5) to water (n=1.33). Find critical angle for total internal reflection.",
    "In Young's double slit experiment, slit separation d=0.5mm, screen distance D=1m, wavelength λ=600nm. Find fringe width and position of 3rd bright fringe.",
    "A diffraction grating has 600 lines/mm. Find the angle of first and second order maxima for λ=500nm. What is the maximum order visible?",
    "A Michelson interferometer uses λ=589nm light. Mirror moves 0.5mm. How many fringes shift? What is the coherence length if fringes vanish after 2×10⁵ fringes?",
    "A thin film of soap (n=1.33) has thickness t=300nm in air. Find wavelengths in visible range (400-700nm) that undergo constructive interference in reflection.",
    "A telescope has objective lens f_o=100cm and eyepiece f_e=5cm. Find angular magnification and length of telescope for object at infinity.",
    "Light of intensity I0 passes through two polarizers with axes at 60°. Find transmitted intensity. If a third polarizer is inserted at 30°, find new intensity.",
    "Calculate the resolving power of a diffraction grating with 5000 lines, in second order. Can it resolve sodium D lines at 589.0nm and 589.6nm?",
    "A glass prism of apex angle A=60° and n=1.5 is at minimum deviation. Find angle of minimum deviation and angle of incidence.",

    # Modern Physics & Quantum
    "Photoelectric effect: work function of metal φ=4.5eV. Find threshold frequency, max KE of electrons for λ=200nm, and stopping potential.",
    "Hydrogen atom: Find energy, radius, and speed of electron in n=3 orbit using Bohr model. What is wavelength of photon emitted for n=3→2 transition?",
    "De Broglie wavelength: Find wavelength of electron accelerated through V=100V and proton through same voltage. Compare with X-ray wavelength.",
    "Compton scattering: X-ray of λ=0.1Å scatters off free electron at θ=90°. Find wavelength shift, scattered wavelength, and recoil energy of electron.",
    "Nuclear decay: ²³⁸U decays with half-life t½=4.5×10⁹ years. Find decay constant, activity of 1g sample, and time for 90% decay.",
    "Nuclear binding energy: Calculate binding energy per nucleon of ⁵⁶Fe given mass=55.9349u, proton mass=1.00728u, neutron mass=1.00866u.",
    "Using Heisenberg uncertainty principle, estimate minimum kinetic energy of electron confined in nucleus (r~10⁻¹⁵m) and in atom (r~10⁻¹⁰m).",
    "A particle in infinite potential well of width L=0.1nm. Find ground state energy, first excited state energy, and wavelength of photon emitted for E2→E1 transition.",
    "Relativistic mechanics: An electron moves at v=0.9c. Find its relativistic mass, kinetic energy, and total energy. Compare KE with classical value.",
    "Pair production: A gamma ray produces electron-positron pair. Find minimum frequency of photon required. If γ energy is 2MeV, find KE of each particle.",

    # Waves & Oscillations
    "A string of length L=2m, mass m=0.01kg is under tension T=50N. Find wave speed, fundamental frequency, and frequencies of first three harmonics.",
    "Two waves y1=A sin(kx-ωt) and y2=A sin(kx-ωt+π/3) superpose. Find amplitude, phase, and intensity of resultant wave.",
    "A sound wave in air (v=340 m/s) and another in water (v=1480 m/s) have same frequency f=1000Hz. Compare wavelengths and find ratio of wave numbers.",
    "Doppler effect: Source emits f=500Hz moving toward observer at 30 m/s. Observer moves away at 10 m/s. Speed of sound=340 m/s. Find observed frequency.",
    "A standing wave y=2A cos(kx)sin(ωt) has A=0.02m, k=π/2 rad/m, ω=100π rad/s. Find nodes, antinodes, wave speed, and max velocity of particle at x=0.25m.",
    "A damped oscillator has m=0.5kg, k=50N/m, b=2 kg/s. Find natural frequency, damped frequency, time constant, and Q-factor.",
    "Beats: Two tuning forks of 440Hz and 444Hz sound together. Find beat frequency and time between successive maxima. What is the resultant frequency?",
    "A pipe of length L=0.85m is open at both ends. Find first three resonant frequencies if speed of sound=340 m/s. Repeat for closed-open pipe.",
    "Longitudinal waves: Find speed of sound in steel (Y=2×10¹¹ Pa, ρ=7800 kg/m³) and compare with speed in air at 300K (γ=1.4, M=29×10⁻³ kg/mol).",
    "Energy in SHM: A mass m=0.2kg on spring k=80N/m has amplitude A=0.1m. Find total energy, max velocity, KE and PE at x=A/2.",

    # Advanced Mechanics
    "Using Lagrangian mechanics, derive equations of motion for a double pendulum. Find normal frequencies for small oscillations.",
    "A gyroscope with angular momentum L=10 kg·m²/s is subjected to a torque τ=2 N·m. Find precession angular velocity and period of precession.",
    "Kepler's laws: A satellite orbits Earth at height h=400km. Find orbital period, speed, and total mechanical energy. (R_E=6400km, M_E=6×10²⁴kg)",
    "Using conservation of angular momentum, explain why a neutron star of radius 10km rotates at 100 rev/s if the original star had radius 10⁷km and rotated once per month.",
    "A rigid body has principal moments of inertia I1=2, I2=3, I3=5 kg·m². It rotates freely with ω=(1,2,1) rad/s. Find kinetic energy and angular momentum.",
    "Elastic collision in CM frame: derive expressions for scattering angle in lab frame as function of CM angle for equal mass particles.",
    "A chain of length L and mass M hangs from a table with fraction f hanging off edge. Find minimum f for chain to slide and time to fall completely off.",
    "Virial theorem: For inverse square law gravitational force, prove ⟨T⟩ = -½⟨V⟩ and find total energy of circular orbit in terms of potential energy.",
    "A rocket at rest fires exhaust at u=2500 m/s relative to rocket. What fraction of initial mass must be fuel to achieve v=7000 m/s (orbital velocity)?",
    "Find the moment of inertia of a solid cone of mass M, height h, base radius R about its symmetry axis and about an axis through apex perpendicular to symmetry axis.",

    # Electrodynamics
    "An LC circuit has L=10mH, C=100μF, initial charge Q0=1mC. Find frequency, max current, max energy in inductor, and charge as function of time.",
    "Maxwell's equations: Write all four in differential and integral form. Derive the wave equation for E and B in vacuum and find wave speed.",
    "A plane EM wave has E=E0 sin(kz-ωt) x̂. Find B field, Poynting vector, radiation pressure, and intensity for E0=100 V/m.",
    "Skin effect: Find skin depth δ for copper (σ=6×10⁷ S/m, μr=1) at frequencies 60Hz, 1MHz, and 10GHz.",
    "A transformer has primary N1=500 turns, secondary N2=50 turns. Primary voltage V1=220V, primary current I1=2A. Find V2, I2, and power assuming 90% efficiency.",
    "Transmission line: A coaxial cable has inner radius a=1mm, outer radius b=5mm, filled with dielectric ε_r=2.3. Find capacitance and inductance per unit length and wave speed.",
    "A circular loop of radius r=10cm in a time-varying field B=B0 sin(ωt) with B0=0.1T, ω=100π rad/s. Find induced EMF and current if R=5Ω.",
    "Hall effect: Copper strip width w=1cm, thickness t=1mm, current I=10A, B=0.5T perpendicular. Find Hall voltage, Hall coefficient, and carrier density.",
    "Radiation from accelerating charge: An electron oscillates with amplitude A=0.1nm and frequency f=10¹⁵Hz. Find radiated power using Larmor formula.",
    "A coaxial capacitor of length L=0.5m, inner radius a=1cm, outer radius b=3cm is filled with dielectric ε_r=4. Find capacitance, charge for V=1000V, and energy stored.",

    # Statistical Mechanics
    "Using Boltzmann distribution, find fraction of molecules in first excited state (E=0.1eV above ground) at T=300K and T=3000K.",
    "Equipartition theorem: Find average kinetic energy, total internal energy of 1 mole of diatomic ideal gas (N2) at T=500K. Include translational and rotational modes.",
    "Fermi-Dirac distribution: Find Fermi energy of copper (n=8.5×10²⁸ m⁻³) and probability of occupation at E_F+0.1eV at T=300K.",
    "Bose-Einstein condensation: Estimate critical temperature for Rb atoms (m=85u) at n=10¹⁴ cm⁻³. What fraction condenses at T=0.5T_c?",
    "Random walk: A particle takes N=10000 steps of length l=1nm in 3D. Find RMS displacement, diffusion coefficient if each step takes τ=10⁻¹²s.",
    "Partition function: Calculate partition function for 2-level system with energies 0 and ε=0.05eV at T=300K. Find average energy and heat capacity.",
    "Maxwell-Boltzmann: Derive expression for speed distribution. Find fraction of N2 molecules at 300K with speeds between 400-500 m/s.",
    "Blackbody radiation: Find peak wavelength (Wien's law), total power radiated per unit area (Stefan-Boltzmann) for T=5778K (Sun's surface).",
    "Entropy of mixing: Two ideal gases N2 and O2, each 1 mole at same T and P, are mixed. Calculate entropy change of mixing.",
    "Phase transition: Water at triple point has T=273.16K, P=611Pa. Using Clausius-Clapeyron, estimate dP/dT for ice-water and water-vapor boundaries.",
]


class SGLangLoadTest(HttpUser):
    wait_time = between(1, 3)
    host = "https://uoum3yyi76wdqh-8000.proxy.runpod.net"

    @task
    def chat_completion(self):
        prompt = random.choice(IRODOV_PROMPTS)
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert physics and mathematics professor. Solve problems step by step with clear reasoning."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
        with self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                catch_response=True,
                timeout=120,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                message = data.get("choices", [{}])[0].get("message", {})
                # Qwen3 may return content in either field depending on thinking mode
                content = message.get("content") or message.get("reasoning_content")
                if content:
                    response.success()
                else:
                    response.failure(f"Empty content in response: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")