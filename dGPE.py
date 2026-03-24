# Simulation du modèle GPE dissipatif

'''
But : Simuler le condensat de polariton + reservoir en résolvant les equations couplées avec la forme dissipative du modèle GPE 
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import subprocess

#Constante
HBAR_SI    = 1.054_571_817e-34   # J·s
QE         = 1.602_176_634e-19   # C
ME_SI      = 9.109_383_7015e-31  # kg
UM_TO_M    = 1e-6                # µm -> m
PS_TO_S    = 1e-12               # ps -> s
HBAR_meVps = (HBAR_SI / QE) * 1e3 / PS_TO_S   # ħ ≈ 0.658211951 meV·ps

def distance_periodique(x, x0, L):
    # Calcul la distance minimale entre x et x0 sur une boite périodique de longueur L
    return ((x - x0 + L/2) % L) - L/2

def potentiel_nul(x):
    #Défini un potentiel nul
    return np.zeros_like(x, dtype=float)

def pompe(x, L, P0, x0, L0, sigma):
    #Profil de pompage plat au centre avec bords lissé par fonction tanh, type flat-top
    dx = distance_periodique(x, x0, L)
    
    #Produit de 2 sigmoides formant un plateau
    prod = (1 + np.tanh((dx + L0)/sigma)) * (1 + np.tanh((L0 - dx)/sigma))

    #Normalisation
    norm = (1 + np.tanh(L0/sigma))**2
    return P0 * prod / norm

def omega_k_ps(k_um, m_rel):
    #Dispersion cinétique en ps-1
    k_SI = k_um / UM_TO_M
    m_SI = m_rel * ME_SI
    omega_s = (HBAR_SI * k_SI**2) / (2.0 * m_SI)
    return omega_s * PS_TO_S

def dGPE(psi0, nR0, # Etat initial
        L_um, # Taille du système en um
        N, # Nombre de points sur la grille
        dt_ps, # Pas de temps en ps
        n_steps, # Nombre d'itérations
        out_every, #Sauvegarde
        m_rel, # masse effective (m/m_e)
        g_meVum, # Intéractions entre polaritons
        gR_meVum, # Intéraction polariton - réservoir
        R_ps_inv, # Taux de scattering
        gamma_c_ps_inv, # Perte du condensat
        gamma_R_ps_inv, #Perte du réservoir
        V_meV, # Potentiel
        P_um_inv_ps_inv, # Pompe
        noise_D_psi, #Amplitude du bruit
        dt_transi, #Pas de temps du transitoire en ps
        n_steps_transi, #Nombres d'itérations du transitoire
        seed):
    #Intégration du système

    rng = np.random.default_rng(seed)

    #Copie de l'état initiale
    psi = psi0.copy().astype(np.complex128)
    nR = nR0.copy().astype(float)

    #Grille de l'espace réel
    x = np.linspace(0, L_um, N)
    dx = x[1] - x[0]

    #Grille de l'espace de Fourier
    k_um = 2*np.pi * np.fft.fftfreq(N, d=dx)

    #Propagateur cinétique
    omega_k = omega_k_ps(k_um, m_rel)

    #Transitoire
    Kprop_coarse = np.exp(-(1j * omega_k) * dt_transi)
    noise_sigma_coarse = np.sqrt(max(0, 2 * noise_D_psi * dt_transi / dx))

    t_ps = 0
    
    for n in range(1, n_steps_transi + 1):
        # Densité au début
        rho_deb = np.abs(psi)**2

        # Énergie locale & fréquence
        local_E_meV = (g_meVum * rho_deb + 2.0 * gR_meVum * nR + V_meV)
        local_omega = local_E_meV / HBAR_meVps

        # Gain linéaire
        gainlin = 0.5 * (R_ps_inv * nR - gamma_c_ps_inv)

        # Demi-pas local
        psi *= np.exp(-1j * local_omega * (dt_transi / 2.0)) \
               * np.exp(gainlin * (dt_transi / 2.0))

        # Pas cinétique
        psi_k = np.fft.fft(psi)
        psi_k *= Kprop_coarse
        psi = np.fft.ifft(psi_k)

        # Densité au milieu
        rho_mid = np.abs(psi)**2

        # Mise à jour nR (RK2)
        k1 = P_um_inv_ps_inv - (gamma_R_ps_inv + R_ps_inv * rho_deb) * nR
        n_half = nR + 0.5 * dt_transi * k1
        k2 = P_um_inv_ps_inv - (gamma_R_ps_inv + R_ps_inv * rho_mid) * n_half
        nR = np.clip(nR + dt_transi * k2, 0.0, None)

        # Deuxième demi-pas
        rho_fin = np.abs(psi)**2
        local_E2_meV = (g_meVum * rho_fin + 2.0 * gR_meVum * nR + V_meV)
        local_omega2 = local_E2_meV / HBAR_meVps
        gainlin2 = 0.5 * (R_ps_inv * nR - gamma_c_ps_inv)
        psi *= np.exp(-1j * local_omega2 * (dt_transi / 2.0)) \
               * np.exp(gainlin2 * (dt_transi / 2.0))

        # Bruit
        if noise_sigma_coarse > 0.0:
            noise = (rng.normal(size=N) + 1j*rng.normal(size=N)) \
                     * (noise_sigma_coarse/np.sqrt(2.0))
            psi += noise

        # Avancement du temps
        t_ps += dt_transi



    Kprop = np.exp(-(1j * omega_k) * dt_ps)

    # Bruit pour dt_fine
    noise_sigma = np.sqrt(max(0.0, 2.0 * noise_D_psi * dt_ps / dx))

    # Listes de stockage : on commence à t_ps (≈ 10 ns)
    psi_list = [psi.copy()]
    nR_list  = [nR.copy()]
    t_list   = [t_ps]

    #Strang split-step + RK2
    for n in range(1, n_steps + 1):
        #Densité de particules a chaque x
        rho_deb = np.abs(psi)**2

        #Energie local
        local_E_meV = (g_meVum * rho_deb + 2 * gR_meVum * nR + V_meV)

        #Conversion énergie en fréquence
        local_omega = local_E_meV / HBAR_meVps

        #Gain linéaire du condensat
        gainlin = 0.5 * (R_ps_inv * nR - gamma_c_ps_inv)

        #Demi-pas local
        psi *= np.exp(-1j * local_omega * (dt_ps / 2)) * np.exp(gainlin * (dt_ps / 2))

        #Pas cinétique
        psi_k = np.fft.fft(psi)
        psi_k *= Kprop
        psi = np.fft.ifft(psi_k)

        #Densité de particule au milieu de l'intégration
        rho_mid = np.abs(psi)**2

        #Intégration avec RK2 de nR
        k1 = P_um_inv_ps_inv - (gamma_R_ps_inv + R_ps_inv * rho_deb) * nR
        n_half = nR + 0.5 * dt_ps * k1
        k2 = P_um_inv_ps_inv - (gamma_R_ps_inv + R_ps_inv * rho_mid) * n_half
        nR = np.clip(nR + dt_ps * k2, 0, None)

        
        rho_fin = np.abs(psi)**2
        local_E2_meV = (g_meVum * rho_fin + 2 * gR_meVum * nR + V_meV)
        local_omega2 = local_E2_meV / HBAR_meVps
        gainlin2 = 0.5 * (R_ps_inv * nR - gamma_c_ps_inv)

        #Demi-pas local
        psi *= np.exp(-1j * local_omega2 * (dt_ps/2)) * np.exp(gainlin2 * (dt_ps/2))

        #Bruit
        if noise_sigma > 0:
            noise = (rng.normal(size=N) + 1j*rng.normal(size=N)) * (noise_sigma/np.sqrt(2))
            psi += noise

        # Avancement du temps
        t_ps += dt_ps

        #Sauvegarde
        if (n % out_every) == 0:
            psi_list.append(psi.copy())
            nR_list.append(nR.copy())
            t_list.append(t_ps)

    return x, k_um, psi_list, nR_list, t_list

def evolution_temporelle(psi_list, nR_list, x, R=None, gamma_c=None):
    #Evolution temporelle de la densité du condensat, du réservoir et du gain.
    Nc = [np.trapezoid(np.abs(psi)**2, x) for psi in psi_list]
    NR = [np.trapezoid(nR, x)            for nR  in nR_list]
    if (R is not None) and (gamma_c is not None):
        G = [np.mean(R * nR - gamma_c) for nR in nR_list]
    else:
        G = None
    return np.array(Nc), np.array(NR), (None if G is None else np.array(G))



### Export animation en mp4 ###

def _script_dir():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def _try_save_mp4(anim, out_path, fps, bitrate, dpi, codecs=("libx264","h264_videotoolbox","mpeg4")):
    for codec in codecs:
        try:
            writer = animation.FFMpegWriter(
                fps=fps, bitrate=bitrate, codec=codec,
                extra_args=["-pix_fmt", "yuv420p"]
            )
            anim.save(out_path, writer=writer, dpi=dpi)
            print(f"[OK] MP4 écrit avec codec={codec} -> {out_path}")
            return out_path
        except subprocess.CalledProcessError as e:
            print(f"[ffmpeg] Échec codec={codec} (code={e.returncode}). On essaie le suivant…")
        except Exception as e:
            print(f"[ffmpeg] Échec codec={codec} ({type(e).__name__}: {e}). On essaie le suivant…")
    raise RuntimeError("Aucun codec MP4 n’a fonctionné.")

def save_line_animation(x, series, t_list, y_label, title, filename, fps=25, dpi=150, bitrate=1800):
    script_dir = _script_dir()
    out_path = script_dir / filename

    y_min = min(float(np.nanmin(y)) for y in series)
    y_max = max(float(np.nanmax(y)) for y in series)
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], linewidth=1.5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x [µm]"); ax.set_ylabel(y_label); ax.set_title(title)

    def init():
        line.set_data([], [])
        time_text.set_text("")
        return (line, time_text)

    def update(i):
        line.set_data(x, series[i])
        time_text.set_text(f"t = {t_list[i]:.3f} ps")
        return (line, time_text)

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(series), interval=1000/fps, blit=True
    )

    try:
        if out_path.suffix.lower() == ".mp4":
            if not animation.writers.is_available("ffmpeg"):
                print("[Avertissement] ffmpeg non détecté -> fallback GIF.")
                out_path = out_path.with_suffix(".gif")
                writer = animation.PillowWriter(fps=fps)
                anim.save(out_path, writer=writer, dpi=dpi)
            else:
                try:
                    _try_save_mp4(anim, out_path, fps, bitrate, dpi)
                except Exception:
                    print("[Avertissement] MP4 impossible. Fallback GIF.")
                    out_path = out_path.with_suffix(".gif")
                    writer = animation.PillowWriter(fps=fps)
                    anim.save(out_path, writer=writer, dpi=dpi)
        else:
            writer = animation.PillowWriter(fps=fps)
            anim.save(out_path, writer=writer, dpi=dpi)
    finally:
        plt.close(fig)

    return out_path






### Simulation ###


# Paramètres
L = 200          # µm
N = 1024
dt = 4e-2 # ps 
n_steps = 1000000
out_every = 100
x = np.linspace(0, L, N, endpoint=False)

# --- Paramètres physiques ---
m_rel = -3.3e-6
g_meVum = 0
gR_meVum = 1
gamma0_meV = 0.0485
gamma_c = gamma0_meV / HBAR_meVps
gamma_R = 0.45 * gamma_c
R_eff = 8.8e-4
noise_D = 1e-8
dt_transi = 2e-3 
n_steps_transi = 1
seed = 42

#Intensité minimale de la pompe nécessaire pour la condensation
P_th = gamma_c * gamma_R / R_eff
P0   = 1.15 * P_th

#Pompe et potentiel
P = pompe(x, L, P0, L/2, 80, 9.7)
V = potentiel_nul(x)

#Etat initiaux
rng = np.random.default_rng(123)
psi0 = 1e-4 * (1 + 0.01*rng.random(N)) * np.exp(1j*0.0*x)
nR0  = P / max(gamma_R, 1e-12)

#Intégration
x, k_um, psi_list, nR_list, t_list = dGPE(psi0, nR0, L, N, dt, n_steps, out_every, m_rel, g_meVum, gR_meVum, R_eff, gamma_c, gamma_R, V, P, noise_D, dt_transi, n_steps_transi, seed)

densite = [np.abs(psi)**2 for psi in psi_list]

def centered_phase(psi):
    #Phase dépliée SANS drift global, enlève la rotation uniforme de phase
    phi = np.unwrap(np.angle(psi))
    return phi - phi.mean()

phase = [centered_phase(psi) for psi in psi_list]

# Récupérer le dossier du script courant
script_dir = Path(__file__).resolve().parent

# Chemin complet pour sauvegarder le fichier
out_path = script_dir / "simulation_dGPE.npz"

# Sauvegarde
np.savez(
    out_path,
    x=x,
    k_um=k_um,
    t_list=np.array(t_list),
    psi_list=np.array(psi_list),
    nR_list=np.array(nR_list),
    m_rel=m_rel)

print(f"Données sauvegardées dans : {out_path}")

'''
# --- Vidéos (MP4 avec fallback GIF) ---
tag = "dGPE"

dens_vid = save_line_animation(
    x, densite, t_list,
    y_label="|ψ|²  [µm⁻¹]",
    title=f"dGPE — densité",
    filename=f"densite_{tag}.mp4", fps=25
); print(f"Vidéo densité : {dens_vid.resolve()}")

phase_vid = save_line_animation(
    x, phase, t_list,
    y_label="φ  [rad]",
    title=f"dGPE — phase",
    filename=f"phase_{tag}.mp4", fps=25
); print(f"Vidéo phase   : {phase_vid.resolve()}")

'''
# Calcul des séries temporelles
Nc, NR, G = evolution_temporelle(psi_list, nR_list, x, R_eff, gamma_c)

# Conversion t_list en tableau numpy
t_array = np.array(t_list)

# --- Plots ---
plt.figure(figsize=(10,4))
plt.plot(t_array, Nc, label="Condensat Nc = ∫|ψ|² dx", linewidth=1.5)
plt.xlabel("t [ps]")
plt.ylabel("Nc")
plt.title("Évolution temporelle du condensat")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t_array, NR, label="Réservoir NR = ∫n_R dx", linewidth=1.5, color="orange")
plt.xlabel("t [ps]")
plt.ylabel("NR")
plt.title("Évolution temporelle du réservoir")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

if G is not None:
    plt.figure(figsize=(10,4))
    plt.plot(t_array, G, label="Gain effectif <R n_R - γ_c>", linewidth=1.5, color="green")
    plt.xlabel("t [ps]")
    plt.ylabel("Gain [ps⁻¹]")
    plt.title("Évolution temporelle du gain effectif")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

