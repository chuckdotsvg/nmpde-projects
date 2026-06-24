#ifndef IONIC_MODEL_HPP
#define IONIC_MODEL_HPP

#include <algorithm>
#include <cmath>

/**
 * @file IonicModel.hpp
 * @brief Implementazione del Modello Minimo Fenomenologico per il potenziale d'azione ventricolare umano.
 * * Basato ESATTAMENTE sulle equazioni e sui parametri (Set Epicardico) della Tabella 1 del paper:
 * "Minimal model for human ventricular action potentials in tissue"
 * Alfonso Bueno-Orovio, Elizabeth M. Cherry, Flavio H. Fenton.
 * Journal of Theoretical Biology 253 (2008) 544-560.
 * * Il modello riduce la complessità dei canali ionici a 3 correnti principali (J_fi, J_so, J_si)
 * governate dal potenziale u e da 3 variabili di gating (v, w, s).
 */

// Stato locale della singola cellula (o nodo della mesh).
// Nel modello di Bueno-Orovio il voltaggio 'u' è adimensionale (scala ~ 0.0 - 1.55).
struct IonicState {
    double u; // Potenziale transmembrana (adimensionale)
    double v; // Gate v (governa la corrente rapida del sodio)
    double w; // Gate w (governa le correnti lente, recupero)
    double s; // Gate s (governa la corrente lenta entrante, plateau)
};

class IonicModel {
public:
    IonicModel() {
        // Parametri per cellula EPICARDICA (Rif: Tabella 1, Bueno-Orovio et al. 2008)
        // Soglie di voltaggio
        u_o = 0.0; 
        u_u = 1.55;             // Picco massimo del potenziale
        theta_v = 0.3;          // Soglia di ATTIVAZIONE per la corrente rapida (Sodio)
        theta_w = 0.13;         // Soglia di ATTIVAZIONE per le correnti lente
        theta_v_minus = 0.006;  // Soglia di pre-allarme/inattivazione per il gate v
        theta_o = 0.006;        // Soglia per l'attivazione della corrente uscente di base

        // Costanti di tempo per il gate v (Sodio)
        tau_v1_minus = 60.0; 
        tau_v2_minus = 1150.0;
        tau_v_plus = 1.4506; 

        // Costanti di tempo per il gate w (Potassio/Recupero)
        tau_w1_minus = 60.0; 
        tau_w2_minus = 15.0;
        k_w_minus = 65.0; 
        u_w_minus = 0.03; 
        tau_w_plus = 200.0;
        tau_w_inf = 0.07;       // Parametro w_inf corretto da Tabella 1
        w_inf_star = 0.94;

        // Costanti di tempo per le correnti (J_fi, J_so)
        tau_fi = 0.11;          // Velocità upstroke (più è piccolo, più è ripido)
        tau_o1 = 400.0; 
        tau_o2 = 6.0;
        tau_so1 = 30.0181;      // Valore esatto da Tabella 1
        tau_so2 = 0.9957; 
        k_so = 2.0458; 
        u_so = 0.65;

        // Costanti di tempo per il gate s (Calcio/Plateau)
        tau_s1 = 2.7342; 
        tau_s2 = 16.0; 
        k_s = 2.0994; 
        u_s = 0.9087;
        tau_si = 1.8875; 
    }

    /**
     * @brief Calcola la somma delle correnti ioniche (RHS dell'equazione Monodominio).
     */
    double compute_total_current(const IonicState &state) const {
        double u = state.u;
        double v = state.v;
        double w = state.w;
        double s = state.s;

        // 1. J_fi: Corrente Rapida Entrante (Sodio). Responsabile dell'upstroke.
        // Si accende SOLO se u supera la soglia theta_v (0.3).
        double H_u_thv = (u > theta_v) ? 1.0 : 0.0;
        double J_fi = -(v * H_u_thv * (u - theta_v) * (u_u - u)) / tau_fi;

        // 2. J_so: Corrente Lenta Uscente (Potassio). Responsabile della ripolarizzazione.
        double H_u_thw = (u > theta_w) ? 1.0 : 0.0;
        double J_so = ((u - u_o) * (1.0 - H_u_thw) / tau_o(u)) + (H_u_thw / tau_so(u));

        // 3. J_si: Corrente Lenta Entrante (Calcio). Sostiene il plateau.
        double J_si = - (H_u_thw * w * s) / tau_si;

        return J_fi + J_so + J_si;
    }

    /**
     * @brief Evolve le variabili di gating nel tempo usando Forward Euler esplicito.
     */
    void evolve_gates(IonicState &state, double dt) const {
        double u = state.u;
        
        // --- Evoluzione GATE V (Sodio) ---
        double H_u_thv_m = (u > theta_v_minus) ? 1.0 : 0.0;
        double v_inf = (u < theta_v_minus) ? 1.0 : 0.0;
        double tau_v_m = (1.0 - H_u_thv_m) * tau_v1_minus + H_u_thv_m * tau_v2_minus;
        
        // L'integrazione di v dipende RIGOROSAMENTE dalla vera soglia di attivazione (theta_v = 0.3)
        double H_u_thv = (u > theta_v) ? 1.0 : 0.0;
        double dv = (1.0 - H_u_thv) * (v_inf - state.v) / tau_v_m - H_u_thv * state.v / tau_v_plus;
        state.v += dt * dv;

        // --- Evoluzione GATE W (Potassio) ---
        double H_u_tho = (u > theta_o) ? 1.0 : 0.0;
        double w_inf = (1.0 - H_u_tho) * (1.0 - (u / tau_w_inf)) + H_u_tho * w_inf_star;
        
        double tau_w_m = tau_w1_minus + (tau_w2_minus - tau_w1_minus) * (1.0 + std::tanh(k_w_minus * (u - u_w_minus))) / 2.0;
        
        // L'integrazione di w dipende RIGOROSAMENTE dalla soglia per correnti lente (theta_w = 0.13)
        double H_u_thw = (u > theta_w) ? 1.0 : 0.0;
        double dw = (1.0 - H_u_thw) * (w_inf - state.w) / tau_w_m - H_u_thw * state.w / tau_w_plus;
        state.w += dt * dw;

        // --- Evoluzione GATE S (Calcio) ---
        // Anche tau_s dipende dalla soglia theta_w (0.13)
        double tau_s = (1.0 - H_u_thw) * tau_s1 + H_u_thw * tau_s2; 
        double ds = (((1.0 + std::tanh(k_s * (u - u_s))) * 0.5) - state.s)/tau_s;
        state.s += dt * ds;
    }

private:
    double u_o, u_u, theta_v, theta_w, theta_v_minus, theta_o;
    double tau_v1_minus, tau_v2_minus, tau_v_plus;
    double tau_w1_minus, tau_w2_minus, k_w_minus, u_w_minus, tau_w_plus;
    double tau_fi, tau_o1, tau_o2, tau_so1, tau_so2, k_so, u_so;
    double tau_s1, tau_s2, k_s, u_s, tau_si;
    double w_inf_star;
    double tau_w_inf; 

    // Funzioni helper per costanti di tempo dipendenti dal voltaggio
    double tau_o(double u) const {
        return (1.0 - (u > theta_o ? 1.0:0.0))*tau_o1 + (u > theta_o ? 1.0:0.0)*tau_o2;
    }

    double tau_so(double u) const {
        return tau_so1 + (tau_so2 - tau_so1)*(1.0 + std::tanh(k_so*(u - u_so)))*0.5;
    }
};

#endif
