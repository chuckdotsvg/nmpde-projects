#ifndef IONIC_MODEL_HPP
#define IONIC_MODEL_HPP

#include <algorithm>
#include <cmath>

// Stato locale del modello ionico di Bueno-Orovio.
// u rappresenta il potenziale transmembrana, mentre v, w e s sono variabili
// di gating che controllano le correnti ioniche e l'evoluzione della cellula.
struct IonicState {
    double u; // Potenziale (adimensionale o mV)
    double v; // Gate v
    double w; // Gate w
    double s; // Gate s
};

class IonicModel {
public:
    IonicModel() {
        // Parametri del modello epicardico presi dalla formulazione originale.
        // I valori definiscono soglie e costanti di tempo per le correnti
        // rapide, lente e per la dinamica dei gate.
        u_o = 0.0; u_u = 1.55; theta_v = 0.3; theta_w = 0.13; theta_v_minus = 0.006;
        theta_o = 0.006; tau_v1_minus = 60.0; tau_v2_minus = 1150.0;
        tau_v_plus = 1.4506; tau_w1_minus = 60.0; tau_w2_minus = 15.0;
        k_w_minus = 65.0; u_w_minus = 0.03; tau_w_plus = 200.0;
        tau_fi = 0.11; tau_o1 = 400.0; tau_o2 = 6.0;
        tau_so1 = 30.045; tau_so2 = 0.9957; k_so = 2.0458; u_so = 0.65;
        tau_s1 = 2.7342; tau_s2 = 16.0; k_s = 2.0994; u_s = 0.9087;
        tau_si = 1.8875; theta_w_plus = 0.006; w_inf_star = 0.94;
    }

    // Calcola la corrente totale usata nel termine di reazione del PDE.
    // La somma include la corrente rapida entrante, la corrente lenta uscente
    // e la corrente lenta entrante.
    double compute_total_current(const IonicState &state) const {
        double u = state.u;
        double v = state.v;
        double w = state.w;
        double s = state.s;

        // Corrente rapida entrante: si attiva sopra la soglia theta_v.
        double H_u_thv = (u > theta_v) ? 1.0 : 0.0;
        double J_fi = (v * H_u_thv * (u - theta_v) * (u_u - u)) / tau_fi;

        // Corrente lenta uscente: usa due regimi, uno sotto theta_w e uno sopra.
        double H_u_thw = (u > theta_w) ? 1.0 : 0.0;
        double J_so = ((u - u_o) * (1.0 - H_u_thw) / tau_o(u)) + (H_u_thw / tau_so(u));

        // Corrente lenta entrante: dipende dai gate w e s quando il potenziale è alto.
        double J_si = - (H_u_thw * w * s) / tau_si;

        return J_fi + J_so + J_si;
    }

    // Aggiorna i gate con uno schema Forward Euler esplicito.
    // Il passo temporale del progetto è piccolo, quindi l'approccio è stabile
    // a livello pratico per la simulazione prevista.
    void evolve_gates(IonicState &state, double dt) const {
        double u = state.u;
        
        // Gate v: controllo della disponibilità della corrente rapida.
        double H_u_thv_m = (u > theta_v_minus) ? 1.0 : 0.0;
        double v_inf = (u < theta_v_minus) ? 1.0 : 0.0;
        double tau_v_m = (1.0 - H_u_thv_m) * tau_v1_minus + H_u_thv_m * tau_v2_minus;
        double dv = (1.0 - H_u_thv_m) * (v_inf - state.v) / tau_v_m - H_u_thv_m * state.v / tau_v_plus;
        state.v += dt * dv;

        // Gate w: governa il recupero e il blocco della dinamica lenta.
        double H_u_thw_m = (u > theta_w_plus) ? 1.0 : 0.0;
        double w_inf = (1.0 - H_u_thw_m) * (1.0 - (u / tau_w_inf)) + H_u_thw_m * w_inf_star;
        double tau_w_m = tau_w1_minus + (tau_w2_minus - tau_w1_minus) * (1.0 + std::tanh(k_w_minus * (u - u_w_minus))) / 2.0;
        double dw = (1.0 - H_u_thw_m) * (w_inf - state.w) / tau_w_m - H_u_thw_m * state.w / tau_w_plus;
        state.w += dt * dw;

        // Gate s: rappresenta il ramo più lento della reazione ionica.
        double tau_s = (1.0 - H_u_thw_m) * tau_s1 + H_u_thw_m * tau_s2; 
        double ds = (((1.0 + std::tanh(k_s * (u - u_s))) * 0.5) - state.s)/tau_s;

        state.s += dt * ds;
    }

private:
    // Parametri del modello e soglie fisiologiche.
    double u_o, u_u, theta_v, theta_w, theta_v_minus, theta_o;
    double tau_v1_minus, tau_v2_minus, tau_v_plus;
    double tau_w1_minus, tau_w2_minus, k_w_minus, u_w_minus, tau_w_plus;
    double tau_fi, tau_o1, tau_o2, tau_so1, tau_so2, k_so, u_so;
    double tau_s1, tau_s2, k_s, u_s, tau_si;
    double theta_w_plus, w_inf_star;
    double tau_w_inf = 1.0; // Scaling factor 

    // Costante di tempo della corrente uscente a basso potenziale.
    double tau_o(double u) const {
        return (1.0 - (u > theta_o ? 1.0:0.0))*tau_o1 + (u > theta_o ? 1.0:0.0)*tau_o2;
    }

    // Costante di tempo della corrente lenta uscente con transizione liscia.
    double tau_so(double u) const {
        return tau_so1 + (tau_so2 - tau_so1)*(1.0 + std::tanh(k_so*(u - u_so)))*0.5;
    }
};

#endif