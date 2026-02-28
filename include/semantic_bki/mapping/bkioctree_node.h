#pragma once

#include <vector>

namespace semantic_bki {

    /// Occupancy state: before pruning: FREE, OCCUPIED, UNKNOWN; after pruning: PRUNED
    enum class State : char {
        FREE, OCCUPIED, UNKNOWN, PRUNED
    };

    /*
     * @brief Inference ouputs and occupancy state.
     *
     * Occupancy has member variables: m_A and m_B (kernel densities of positive
     * and negative class, respectively) and State.
     * Before using this class, set the static member variables first.
     */
    class Semantics {
      
      friend class SemanticBKIOctoMap;

    public:
        /*
         * @brief Constructors and destructor.
         */
        Semantics() : ms(std::vector<float>(num_class, prior)), state(State::UNKNOWN),
                      osm_building(0.f), osm_road(0.f), osm_grassland(0.f), osm_tree(0.f), osm_parking(0.f), osm_fence(0.f), osm_stairs(0.f) { classified = false; }

        Semantics(const Semantics &other) : ms(other.ms), state(other.state), semantics(other.semantics),
                      osm_building(other.osm_building), osm_road(other.osm_road), osm_grassland(other.osm_grassland), osm_tree(other.osm_tree), osm_parking(other.osm_parking), osm_fence(other.osm_fence), osm_stairs(other.osm_stairs) { }

        Semantics &operator=(const Semantics &other) {
          ms = other.ms;
          state = other.state;
          semantics = other.semantics;
          osm_building = other.osm_building;
          osm_road = other.osm_road;
          osm_grassland = other.osm_grassland;
          osm_tree = other.osm_tree;
          osm_parking = other.osm_parking;
          osm_fence = other.osm_fence;
          osm_stairs = other.osm_stairs;
          return *this;
        }

        ~Semantics() { }

        /*
         * @brief Exact updates for nonparametric Bayesian kernel inference
         * @param ybar kernel density estimate of positive class (occupied)
         * @param kbar kernel density of negative class (unoccupied)
         */
        void update(std::vector<float>& ybars);

        /// Get probability of occupancy.
        void get_probs(std::vector<float>& probs) const;

        /// Get variance of occupancy (uncertainty)
	      void get_vars(std::vector<float>& vars) const;
        
        /*
         * @brief Get occupancy state of the node.
         * @return occupancy state (see State).
         */
        inline State get_state() const { return state; }

        inline int get_semantics() const { return semantics; }

        /// OSM prior values in [0,1]: building, road, grassland, tree, parking, fence, stairs (Euclidean signed-distance-based).
        inline float get_osm_building() const { return osm_building; }
        inline float get_osm_road() const { return osm_road; }
        inline float get_osm_grassland() const { return osm_grassland; }
        inline float get_osm_tree() const { return osm_tree; }
        inline float get_osm_parking() const { return osm_parking; }
        inline float get_osm_fence() const { return osm_fence; }
        inline float get_osm_stairs() const { return osm_stairs; }
        inline void set_osm_building(float v) { osm_building = v; }
        inline void set_osm_road(float v) { osm_road = v; }
        inline void set_osm_grassland(float v) { osm_grassland = v; }
        inline void set_osm_tree(float v) { osm_tree = v; }
        inline void set_osm_parking(float v) { osm_parking = v; }
        inline void set_osm_fence(float v) { osm_fence = v; }
        inline void set_osm_stairs(float v) { osm_stairs = v; }

        bool classified;

    private:
        std::vector<float> ms;
        State state;
        int semantics;
        float osm_building;
        float osm_road;
        float osm_grassland;
        float osm_tree;
        float osm_parking;
        float osm_fence;
        float osm_stairs;
        static int num_class;   // number of classes
        
        static float sf2;
        static float ell;   // length-scale
        static float prior; // prior on each class

        static float var_thresh;
        static float free_thresh;     // FREE occupancy threshold
        static float occupied_thresh; // OCCUPIED occupancy threshold
    };

    typedef Semantics SemanticOcTreeNode;
}
