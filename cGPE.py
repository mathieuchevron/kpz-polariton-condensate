# Simulation du modèle GPE conservatif

'''
But : Simuler le condensat de polariton avec une forme du modèle GPE plus simple a intégré grâce a la conservation du nombre de polariton 
et possibilité d'ajouter un potentiel
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import subprocess

def distance_periodique(x, x0, L):
    # Calcul la distance minimale entre x et x0 sur une boite périodique de longueur L
    return ((x - x0 + L/2) % L) - L/2

def potentiel_nul(x):
    #Défini un potentiel nul
    return np.zeros_like(x, dtype=float)

def cGPE(psi0, L, g, V, dt, n_step, out):
    #Intégration de la cGPE

    #Copie de l'état initiale en 128 bits pour une meilleur précision
    psi = psi0.copy().astype(np.complex128)

    #Grille de l'espace réel
    N = psi.size
    dx = L / N
    x = np.arange(N) * dx

    #Grille de l'espace de Fourier
    k = 2*np.pi * np.fft.fftfreq(N, d=dx)

    #Propagateur cinétique
    Kprop = np.exp(-1j * (k**2) * dt)

    #Stockage
    psi_list = [psi.copy()]
    t_list = [0.0]

    #Strang split-step
    for n in range(1, n_step + 1):
        #Demi-pas local
        psi *= np.exp(-1j * (g * np.abs(psi)**2 + V) * (dt / 2))

        #Pas cinétique
        psi_k = np.fft.fft(psi)
        psi_k *= Kprop
        psi = np.fft.ifft(psi_k)

        #Demi-pas local
        psi *= np.exp(-1j * (g * np.abs(psi)**2 + V) * (dt / 2))

        #Sauvegarde
        if (n % out) == 0:
            psi_list.append(psi.copy())
            t_list.append(n * dt)

    return x, k, psi_list, t_list


def conservation(psi, x, V, g, L):
    #Calcul les grandeurs conserver pour voir la qualité de l'intégration
    Np = psi.size
    dx = L / Np
    kvec = 2 * np.pi * np.fft.fftfreq(Np, d=dx)
    psi_k = np.fft.fft(psi)
    dpsi_dx = np.fft.ifft(1j * kvec * psi_k)

    Norme = np.trapezoid(np.abs(psi)**2, x)
    E_cin = np.trapezoid(np.abs(dpsi_dx)**2, x)
    E_int = 0.5 * g * np.trapezoid(np.abs(psi)**4, x)
    E_pot = np.trapezoid(V * np.abs(psi)**2, x)
    E_tot = E_cin + E_int + E_pot
    return Norme, E_cin, E_int, E_pot, E_tot


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


#Paramètres
N = 1024          # points spatiaux (puissance de 2 → FFT rapide)
L = 200.0         # longueur du domaine (période)
g = 1.0           # non-linéarité (g>0 : répulsif)
dt = 0.0025       # pas de temps
n_step = 100000    # nombre de pas (durée T = n_steps * dt)
out = 100    # cadence des instantanés pour video

x = np.linspace(0, L, N, endpoint=False)
V = potentiel_nul(x)

#Conditions initiales
densite0 = 1.0
#Bosse de phase gaussienne centrée en L/2 
phase_bump = 0.2 * np.exp(-((x - L / 2) ** 2) / (2 * 3.0 ** 2))
psi0 = np.sqrt(densite0) * np.exp(1j * phase_bump)

#Intégration
x, k, psi_list, t_list = cGPE(psi0, L, g, V, dt, n_step, out)

densite = [np.abs(psi) ** 2 for psi in psi_list]

def phase_centre(psi):
    #Phase dépliée SANS drift global, enlève la rotation uniforme de phase
    phi = np.unwrap(np.angle(psi))
    return phi - phi.mean()

phase = [phase_centre(psi) for psi in psi_list]

psi_init = psi_list[0]
Normei, E_cini, E_inti, E_poti, E_toti = conservation(psi_init, x, V, g, L)
psi_final = psi_list[-1]
Norme, E_cin, E_int, E_pot, E_tot = conservation(psi_final, x, V, g, L)

print("[Nombre]")
print(f"Nombre initiale : {Normei:6f}")
print(f"Nombre final : {Norme:.6f}")
print(f"Différence nombre : {Norme-Normei:6f}")
print("[Energie]")
print(f"Energies initiale : Cin={E_cini:.6f}  Int={E_inti:.6f}  Pot={E_poti:.6f}  Tot={E_toti:.6f}")
print(f"Energies finale : Cin={E_cin:.6f}  Int={E_int:.6f}  Pot={E_pot:.6f}  Tot={E_tot:.6f}")
print(f"Différence énergie : dCin={E_cin-E_cini:.6f}  dInt={E_int-E_inti:.6f}  dPot={E_pot-E_poti:.6f}  dTot={E_tot-E_toti:.6f}")

'''
# --- Vidéos (MP4 avec fallback GIF) ---
tag = "cGPE"

dens_vid = save_line_animation(
    x, densite, t_list,
    y_label="|ψ|²  [µm⁻¹]",
    title=f"cGPE — densité",
    filename=f"densite_{tag}.mp4", fps=25
); print(f"Vidéo densité : {dens_vid.resolve()}")

phase_vid = save_line_animation(
    x, phase, t_list,
    y_label="φ  [rad]",
    title=f"cGPE — phase",
    filename=f"phase_{tag}.mp4", fps=25
); print(f"Vidéo phase   : {phase_vid.resolve()}")
'''
