import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
import io
import re # Untuk parsing input teks

# --- 1. Fungsi Heuristik (Diganti dengan Input Kustom) ---
def heuristic(node, goal, heuristics_map):
    """Mengambil nilai h(n) dari peta heuristik yang diberikan oleh pengguna."""
    return heuristics_map.get(node, 0) # Default 0 jika node tidak ada di tabel h(n)

# --- 2. Algoritma A* Search dengan Log Lengkap (Disempurnakan) ---
def a_star_search_logged(graph, heuristics_map, start, goal):
    
    # Validasi: Pastikan start dan goal ada di peta heuristik
    if start not in heuristics_map or goal not in heuristics_map:
        return None, 0, []

    # Priority Queue: (f_cost, g_cost, node, path)
    initial_h = heuristic(start, goal, heuristics_map)
    frontier = [(initial_h, 0, start, [start])] 
    cost_so_far = {start: 0}
    step_log = [] 
    
    while frontier:
        f_cost, g_cost, current_node, current_path = heapq.heappop(frontier)
        
        # Log: Mengambil status Open/Closed List saat ini
        open_list_nodes = [n[2] for n in frontier]
        closed_list_nodes = list(sorted([n for n in cost_so_far if n not in open_list_nodes and n != current_node]))
        
        # Log: Node saat ini sedang dieksplorasi (Pop dari Frontier)
        step_log.append({
            'type': 'explore',
            'node': current_node,
            'g': g_cost,
            'h': heuristic(current_node, goal, heuristics_map),
            'f': f_cost,
            'path_so_far': current_path,
            'open_list': list(sorted(open_list_nodes)),
            'closed_list': closed_list_nodes
        })

        if current_node == goal:
            # Log: Tujuan tercapai
            return current_path, g_cost, step_log

        # Eksplorasi tetangga
        for next_node in graph.neighbors(current_node):
            weight = graph[current_node][next_node]['weight']
            new_g_cost = cost_so_far[current_node] + weight
            
            if next_node not in cost_so_far or new_g_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_g_cost
                h_cost = heuristic(next_node, goal, heuristics_map)
                new_f_cost = new_g_cost + h_cost
                new_path = current_path + [next_node]
                
                heapq.heappush(frontier, (new_f_cost, new_g_cost, next_node, new_path))
                
    return None, 0, step_log

# --- 3. Fungsi Parsing Input Pengguna ---

def parse_input(heuristics_text, edges_text):
    """Menguraikan input teks pengguna menjadi dictionary heuristik dan list edges."""
    
    # 1. Parsing Heuristik (Tabel h(n))
    heuristics_map = {}
    try:
        lines = heuristics_text.strip().split('\n')
        for line in lines:
            parts = re.split(r'[,\s]+', line.strip()) # Pisahkan berdasarkan koma atau spasi
            if len(parts) == 2:
                node, h_val = parts
                heuristics_map[node.upper()] = int(h_val)
    except Exception:
        return None, None, "Format Heuristik tidak valid. Gunakan format 'NODE, NILAI' per baris."

    # 2. Parsing Edge (Jalur dan Jarak)
    edges = []
    try:
        lines = edges_text.strip().split('\n')
        for line in lines:
            parts = re.split(r'[,\s]+', line.strip())
            if len(parts) == 3:
                node1, node2, weight = parts
                edges.append((node1.upper(), node2.upper(), int(weight)))
    except Exception:
        return None, None, "Format Jalur tidak valid. Gunakan format 'NODE1, NODE2, Jarak' per baris."
        
    return heuristics_map, edges, None

# --- 4. Fungsi Visualisasi Graf Langkah demi Langkah ---
def draw_graph_step(G, pos, log, step_number, heuristics_map, path_nodes=None):
    
    plt.figure(figsize=(10, 6))
    
    node_colors = ['skyblue'] * len(G.nodes)
    closed_nodes = log.get('closed_list', [])
    current_node = log.get('node')
    open_nodes = log.get('open_list', [])

    # Update warna node berdasarkan status
    node_color_map = {}
    for i, node in enumerate(G.nodes):
        if node == current_node:
            node_color_map[node] = 'red'  # Node saat ini
        elif node == 'G' or node == st.session_state.goal_node.upper():
            node_color_map[node] = 'green' # Tujuan
        elif node in closed_nodes:
            node_color_map[node] = 'lightcoral' # Closed List
        elif node in open_nodes:
            node_color_map[node] = 'lightgreen' # Open List
        else:
            node_color_map[node] = 'skyblue'
        
        node_colors[i] = node_color_map[node]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)
    nx.draw_networkx_edges(G, pos, edge_color='gray')

    # Sorot Path
    if log.get('type') == 'goal' and path_nodes:
        path_edges = list(zip(path_nodes, path_nodes[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)

    # Label Node
    node_labels = {node: f"{node}\nh={heuristics_map.get(node, '?')}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Label Edge (Biaya/Weight)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(f"Langkah {step_number}: Node Dieksplorasi = {current_node}")
    plt.axis('off')
    
    # Tampilkan plot di Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf.getvalue(), caption=f"Visualisasi Graf - Langkah {step_number}", use_column_width=True)
    plt.close()

# --- 5. Aplikasi Streamlit Utama ---

st.set_page_config(layout="wide")
st.title("⭐️ A\* Search Fleksibel (Input Graf Kustom)")
st.markdown("Masukkan data heuristik dan jalur untuk menghitung dan memvisualisasikan jalur A\*.")

col_input, col_view = st.columns([1, 2.5])

# --- Input Section ---
with col_input:
    st.header("⚙️ Konfigurasi Soal")
    
    # Input Node Start/Goal
    st.session_state.start_node = st.text_input("Node Mulai (e.g., S):", "S").upper()
    st.session_state.goal_node = st.text_input("Node Tujuan (e.g., G):", "G").upper()
    
    st.subheader("1. Tabel Heuristik (h(n))")
    heuristics_text = st.text_area(
        "Format: NODE, NILAI_HEURISTIK (per baris)\nContoh: S, 80",
        "S, 80\nA, 80\nB, 70\nC, 70\nE, 75\nF, 78\nG, 0\nH, 70\nM, 70", height=150
    )
    
    st.subheader("2. Jalur dan Jarak (g(n))")
    edges_text = st.text_area(
        "Format: NODE1, NODE2, JARAK (per baris)\nContoh: S, A, 10",
        "S, A, 10\nS, B, 25\nS, C, 10\nA, G, 105\nB, E, 30\nC, H, 20\nC, F, 5\nE, G, 40\nH, E, 15\nH, M, 40\nF, H, 5\nM, G, 20", height=200
    )
    
    if st.button("Mulai Simulasi A*"):
        heuristics_map, edges, error = parse_input(heuristics_text, edges_text)
        
        if error:
            st.error(f"Kesalahan Input: {error}")
            st.stop()
        
        # Buat Graf NetworkX
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        
        # Hitung posisi node secara dinamis menggunakan spring_layout
        # Ini memastikan graf akan beradaptasi dengan semua bentuk input
        pos = nx.spring_layout(G, k=0.5, iterations=50) 
        
        # Jalankan Algoritma
        path, total_cost, step_log = a_star_search_logged(
            G, heuristics_map, st.session_state.start_node, st.session_state.goal_node
        )
        
        # Simpan hasil untuk visualisasi
        st.session_state.results = {
            'G': G, 'pos': pos, 'h_map': heuristics_map, 
            'path': path, 'total_cost': total_cost, 'log': step_log
        }
    
# --- Visualisasi Section ---
with col_view:
    st.header("🔬 Visualisasi Langkah Per Langkah")
    
    if 'results' in st.session_state and st.session_state.results['log']:
        results = st.session_state.results
        
        # Containers
        graph_container = st.empty()
        list_container = st.empty()
        
        if results['path']:
            st.success(f"🎉 Jalur Ditemukan! Total Biaya Aktual (g(G)): **{results['total_cost']}**")
            st.code(f"Jalur Terpendek: {' -> '.join(results['path'])}")
        else:
            st.error("Jalur tidak ditemukan!")
            
        st.markdown("---")
        
        # --- Loop Simulasi Langkah demi Langkah ---
        for i, log in enumerate(results['log']):
            
            # 1. Visualisasi Graf
            with graph_container.container():
                draw_graph_step(results['G'], results['pos'], log, i + 1, results['h_map'], path_nodes=results['path'] if log['type'] == 'goal' else None)

            # 2. Tampilkan Open/Closed List dan Detail Biaya
            with list_container.container():
                st.subheader(f"Status Daftar (Langkah {i + 1})")
                
                # Open List
                st.markdown(f"**Open List (Frontier):** {', '.join(log['open_list'])}")

                # Closed List
                st.markdown(f"**Closed List (Dieksplorasi):** {', '.join(log['closed_list'])}")
                
                # Tabel Detail Biaya Node Saat Ini
                detail_data = {
                    'Biaya': ['g(n) (Aktual)', 'h(n) (Heuristik)', 'f(n) (Total)'],
                    'Nilai': [log['g'], log['h'], log['f']]
                }
                st.dataframe(pd.DataFrame(detail_data).set_index('Biaya'), use_container_width=True)

            if log['type'] == 'goal':
                st.balloons()
                break
                
            time.sleep(1) 
    else:
        st.info("Masukkan data graf dan heuristik di kolom kiri, lalu tekan 'Mulai Simulasi A*'.")
