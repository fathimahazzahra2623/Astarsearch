import streamlit as st
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import time
import io # Untuk menyimpan plot matplotlib

# --- Data Graf Sesuai Gambar Anda ---
# Format: (node, heuristic value)
HEURISTICS = {
    'S': 80, 'A': 80, 'B': 70, 'C': 70, 'E': 75, 
    'F': 78, 'G': 0, 'H': 70, 'M': 70
}

# Format: (node1, node2, weight)
EDGES = [
    ('S', 'A', 10), ('S', 'B', 25), ('S', 'C', 10),
    ('A', 'G', 105), ('B', 'E', 30), ('C', 'H', 20),
    ('C', 'F', 5), ('E', 'G', 40), ('H', 'E', 15),
    ('H', 'M', 40), ('F', 'H', 5), ('M', 'G', 20)
]

# --- 1. Algoritma A* Search dengan Log Lengkap ---
def a_star_search_logged(graph, start, goal):
    
    frontier = [(HEURISTICS[start], 0, start, [start])] # (f_cost, g_cost, node, path)
    cost_so_far = {start: 0}
    
    # Log utama untuk visualisasi langkah per langkah
    step_log = [] 
    
    while frontier:
        f_cost, g_cost, current_node, current_path = heapq.heappop(frontier)
        
        # Log Open/Closed List Status SEBELUM eksplorasi
        open_list_nodes = [n[2] for n in frontier]
        
        # Log: Node saat ini sedang dieksplorasi (Pop dari Frontier)
        step_log.append({
            'type': 'explore',
            'node': current_node,
            'g': g_cost,
            'h': HEURISTICS.get(current_node, 0),
            'f': f_cost,
            'path_so_far': current_path,
            'open_list': list(sorted(open_list_nodes)), # Node di frontier
            'closed_list': list(sorted([n for n in cost_so_far if n not in open_list_nodes and n != current_node])) # Node yang cost-nya diketahui tapi sudah dieksplorasi/belum di frontier
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
                h_cost = HEURISTICS.get(next_node, 0)
                new_f_cost = new_g_cost + h_cost
                new_path = current_path + [next_node]
                
                heapq.heappush(frontier, (new_f_cost, new_g_cost, next_node, new_path))
                
    return None, 0, step_log # Tidak ditemukan jalur

# --- 2. Fungsi Visualisasi Graf Langkah demi Langkah ---
def draw_graph_step(G, pos, log, step_number, path_nodes=None, current_node=None):
    
    plt.figure(figsize=(10, 6))
    
    # Warna default untuk semua node/edge
    node_colors = ['skyblue'] * len(G.nodes)
    edge_colors = ['gray'] * len(G.edges)
    
    # Node dan Edge yang sudah dieksplorasi (Closed List)
    closed_nodes = log.get('closed_list', [])
    
    # Node yang sedang dieksplorasi (Current Node)
    current_node = log.get('node')

    # Node yang ada di Frontier (Open List)
    open_nodes = log.get('open_list', [])

    # Update warna node berdasarkan status
    node_color_map = {}
    for i, node in enumerate(G.nodes):
        if node == current_node:
            node_color_map[node] = 'red'  # Node saat ini
        elif node == 'G':
            node_color_map[node] = 'green' # Tujuan
        elif node in closed_nodes:
            node_color_map[node] = 'lightcoral' # Closed List
        elif node in open_nodes:
            node_color_map[node] = 'lightgreen' # Open List
        else:
            node_color_map[node] = 'skyblue' # Default
        
        node_colors[i] = node_color_map[node]

    # Gambar Node
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200)

    # Gambar Edge
    nx.draw_networkx_edges(G, pos, edge_color='gray')

    # Sorot Path (Jika Tujuan Sudah Ditemukan)
    if log.get('type') == 'goal' and path_nodes:
        path_edges = list(zip(path_nodes, path_nodes[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)

    # Label Node
    node_labels = {node: f"{node}\nh={HEURISTICS[node]}" for node in G.nodes}
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
    plt.close() # Tutup plot agar memori tidak bocor

# --- 3. Aplikasi Streamlit Utama ---

st.set_page_config(layout="wide")
st.title("⭐️ Visualisasi A* Search pada Graf Kustom")
st.markdown("Visualisasi ini meniru struktur graf dan heuristik dari contoh gambar Anda. ")

# Buat Graf NetworkX
G = nx.Graph()
G.add_weighted_edges_from(EDGES)

# Atur tata letak node secara manual agar mirip gambar (Posisi tetap)
pos = {
    'S': (0, 3), 'A': (2, 4), 'B': (2.5, 3), 'C': (1, 1.5), 
    'E': (4, 3), 'F': (2, 0.5), 'H': (3.5, 1.5), 'M': (5, 1), 'G': (6, 2.5)
}

# Tampilkan Tabel Heuristik Awal
st.subheader("Tabel Heuristik (h(n))")
st.dataframe(pd.DataFrame([HEURISTICS], index=['h(n)']).T, use_container_width=True)
st.markdown("---")

if st.button("Mulai Simulasi A*"):
    path, total_cost, step_log = a_star_search_logged(G, 'S', 'G')

    # Container untuk status graf
    graph_container = st.empty()
    
    # Container untuk open/closed list
    list_container = st.empty()
    
    if path:
        st.success(f"🎉 Jalur Ditemukan! Total Biaya Aktual (g(G)): **{total_cost}**")
        st.code(f"Jalur Terpendek: {' -> '.join(path)}")
    else:
        st.error("Jalur tidak ditemukan!")
        
    st.markdown("---")
    
    # --- Loop Simulasi Langkah demi Langkah ---
    for i, log in enumerate(step_log):
        
        # 1. Visualisasi Graf
        with graph_container.container():
            # Tampilkan graf dengan sorotan node yang sedang dieksplorasi
            draw_graph_step(G, pos, log, i + 1, path_nodes=path if log['type'] == 'goal' else None, current_node=log['node'])

        # 2. Tampilkan Open/Closed List dan Detail Biaya
        with list_container.container():
            st.subheader(f"Status Daftar (Langkah {i + 1})")
            
            # Tabel Open List
            st.info(f"**Node Dieksplorasi (Current):** {log['node']}")

            # Tabel Open List (Frontier)
            open_list_display = pd.DataFrame(log['open_list'], columns=['Node'])
            st.markdown(f"**Open List (Frontier):** {', '.join(log['open_list'])}")

            # Tabel Closed List
            closed_list_display = pd.DataFrame(log['closed_list'], columns=['Node'])
            st.markdown(f"**Closed List:** {', '.join(log['closed_list'])}")
            
            # Tabel Detail Biaya Node Saat Ini
            detail_data = {
                'Biaya': ['g(n) (Aktual)', 'h(n) (Heuristik)', 'f(n) (Total)'],
                'Nilai': [log['g'], log['h'], log['f']]
            }
            st.dataframe(pd.DataFrame(detail_data).set_index('Biaya'), use_container_width=True)


        if log['type'] == 'goal':
            st.balloons()
            break
            
        time.sleep(1) # Jeda visual
