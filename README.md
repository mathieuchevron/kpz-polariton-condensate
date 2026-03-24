# kpz-polariton-condensate
Numerical simulation of KPZ dynamics in a 1D polariton condensate

## PRESENTATION ##

Dans le cadre du projet "Equation KPZ dans un condensat de polariton 1D", les fichiers suivants ont pour but de simuler un condensat de polariton 1D à partir d'une équation de Gross-Pitaevskii (GPE)
et de voir sous quelles conditions, l'évolution d'un condensat de polariton appartient à la classe d'universalité KPZ. 


## cGPE.py ##

-- Description --

1/ Intègre la forme conservative de l'équation de Gross-Pitaevskii, qui a pour but de tester le schéma d'intégration Split-Step Fourier de Strang sur un modèle relativement simple.
2/ Calcule les quantités conservées pour voir la qualité de l'intégration
3/ Enregistre les données sous forme de .mp4

-- Instructions --

- Aucune fonction n'est à modifier
- Choisir ses paramètres / conditions initiales au début de la partie "### Simulation ###"


## dGPE.py ##

-- Description --

1/ Intègre la forme dissipative de l'équation de Gross-Pitaevskii avec le schéma d'intégration Split-Step Fourier de Strang et intègre l'EDO qui régit la densité du réservoir d'excitons non condensés avec le schéma RK2.
2/ Les unités sont : énergie en meV, temps en ps, espace en µm.
3/ Intègre un certain temps sans enregistrer de donnée pour attendre le régime stationnaire et ensuite enregistre les données
4/ Trace les évolutions temporelles de certains paramètres pour voir le régime stationnaire
5/ Enregistre les données sous forme de .mp4
6/ Enregistre différents paramètres sous forme de .npz

-- Instructions --

- Aucune fonction n'est a modifier
- Choisir ses paramètres / conditions initiales au début de la partie "### Simulation ###"
- Par défaut, le script est configuré selon des paramètres à l'article 'Kardar–Parisi–Zhang universality in a one-dimensional polariton condensate'


## JUPY_Analyse.ipynb ##

!!! ATTENTION !!!

Ce script est un notebook jupyter, ce qui est utile pour l'importation des données et leur exploitation sans avoir à tout recompiler

!!! ATTENTION 2 !!!

Ce script ne fonctionne qu'en présence du fichier 'simulation_dGPE.npz', et nécessite donc la compilation du fichier dGPE.py au préalable.

-- Description --

1/ Importe les données du fichier 'simulation_dGPE.npz'
2/ Calcule les fonctions de corrélation de premier ordre g1(Δx,0) et g1(0,Δt) à partir de la phase.
3/ Trace les graphiques de corrélation spatiale et temporelle ainsi que les exposants universels KPZ alpha et beta et le fit des g1 dans un domaine précis

-- Instructions --

- Aucune fonction n'est à modifier
- Définir des intervalles [x_min; x_max] et [t_min; t_max] dans les fonctions de plot
- Ajuster la position des courbes théoriques des exposants universels avec shift_y
- Définir des intervalles [fit_xmin; fit_xmax] et [fit_tmin; fit_tmax] dans les fonctions de plot pour faire un plot sur l'intervalle souhaité
- Si on ne veut pas de fit => fit=False


### KPZ.ipynb ###

!!! ATTENTION !!!

Ce script est un notebook jupyter, ce qui est utile pour l'acquisition des données et leur exploitation sans avoir à tout recompiler

-- Description --

1/ Intègre l'équation KPZ avec le schéma d'intégration Split-Step Fourier de Strang
2/ Calcule les fonctions de corrélation de premier ordre g1(Δx,0) et g1(0,Δt) à partir de la phase.
3/ Trace les graphiques de corrélation spatiale et temporelle ainsi que les exposants universels KPZ alpha et beta et le fit des g1 dans un domaine précis

-- Instructions --

- Aucune fonction n'est à modifier
- Définir les coefficient KPZ, état initial et autres paramètres pour la première cellule
- Définir des intervalles [x_min; x_max] et [t_min; t_max] dans les fonctions de plot
- Ajuster la position des courbes théoriques des exposants universels avec shift_y
- Définir des intervalles [fit_xmin; fit_xmax] et [fit_tmin; fit_tmax] dans les fonctions de plot pour faire un plot sur l'intervalle souhaité
- Si on ne veut pas de fit => fit=False
