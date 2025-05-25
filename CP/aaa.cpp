#include <bits/stdc++.h>
 
// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
 
// #define ordered_set tree<int, null_type, less<int>, rb_tree_tag, tree_order_statistics_node_update>
 
using namespace std;
typedef long long ll;
 
#define int ll
#define ull unsigned long long
#define pii pair<int, int>
#define vi vector<int>
#define vvi vector<vi>
#define vpi vector<pii>
#define vs vector<string>
#define vvs vector<vs>
#define all(vec) (vec).begin(), (vec).end()
#define endl '\n'
#define sp " "
#define printvec(vec)    \
    for (auto i : (vec)) \
        cout << i << sp; \
    cout << endl
 
vector<vector<pair<int, int>>> adj(100005);
vector<bool> vis(2e5 + 10, false);
 
void solveit()
{
    int n, m;
    cin >> n >> m;
 
    for (int i = 0; i < m; i++)
    {
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }
 
    for (int i = 1; i <= n; i++)
    {
        if (adj[i].size() & 1)
        {
            cout << "IMPOSSIBLE\n";
            return;
        }
    }
 
    stack<int> st;
    st.push(1);
 
    vi path;
 
    while (!st.empty())
    {
        int v = st.top();
        int f = 0;
        while (!adj[v].empty())
        {
            auto kk = adj[v].back();
            adj[v].pop_back();
            auto u = kk.first, i = kk.second;
 
            if (!vis[i])
            {
                st.push(u);
                vis[i] = 1;
                f = 1;
                break;
            }
        }
        if (!f)
        {
            path.push_back(v);
            st.pop();
        }
    }
    if (path.size() != m + 1)
    {
        cout << "IMPOSSIBLE\n";
        return;
    }
 
    printvec(path);
}
 
int32_t main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
 
    // freopen ("in", "r", stdin);
    // freopen ("out", "w", stdout);
 
    int t = 1;
    // cin >> t;
 
    for (int tt = 1; tt <= t; tt++)
    {
        // cout << "Case " << tt << ": ";
        solveit();
    }
}