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

void solveit()
{
    int n; cin >> n;
    vpi vec(n);
    
    for (auto &i : vec) cin >> i.first >> i.second;

    map<int, int> mp;

    for (auto i:vec)
    {
        mp[i.first]++;
        mp[-i.second]++;
    }

    int id = 0;
    for (auto &i : mp)
    {
        i.second = id++;
    }

    vector<vector<pair<int, int>>> g(mp.size());
    map<pair<int, int>, int> edge;

    for (int i=0; i<n; i++)
    {
        int u = vec[i].first;
        int v = vec[i].second;

        edge[{mp[u], mp[-v]}] = i+1;
        edge[{mp[-v], mp[u]}] = i+1;

        g[mp[u]].push_back({mp[-v], i+1});
        g[mp[-v]].push_back({mp[u], i+1});
    }

    int start = -1, odd = 0;

    for (int i=0; i<mp.size(); i++)
    {
        if (g[i].size() & 1)
        {
            start = i;
            odd++;
        }
    }

    if ((odd & 1) || odd > 2)
    {
        cout << "No\n";
        return;
    }

    if (start == -1) 
        start = 0;

    stack<int> st;
    st.push(start);

    vi path, idx;
    vi vis(n+1, false);

    while(!st.empty())
    {
        int v = st.top();
        int f = 0;

        while (!g[v].empty())
        {
            auto kk = g[v].back();
            g[v].pop_back();
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

    if (path.size() != n+1)
    {
        cout << "No\n";
        return;
    }

    cout << "Yes\n";

    // printvec(path);

    vi anspath;

    for (int i=1; i<path.size(); i++)
    {
        anspath.push_back(edge[{path[i-1], path[i]}]);
    }

    printvec(anspath);
}

int32_t main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    // freopen ("in", "r", stdin);
    // freopen ("out", "w", stdout);

    int t = 1;
    cin >> t;

    for (int tt = 1; tt <= t; tt++)
    {
        // cout << "Case " << tt << ": ";
        solveit();
    }
}