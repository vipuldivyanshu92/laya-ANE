import SwiftUI
import UIKit

struct RootView: View {
    @EnvironmentObject var engine: LayaEngine

    var body: some View {
        TabView {
            TriageView()
                .tabItem { Label("Inbox", systemImage: "tray.full") }
            GuardView()
                .tabItem { Label("Live Guard", systemImage: "shield.lefthalf.filled") }
            SnakeView()
                .tabItem { Label("Snake", systemImage: "hare") }
            AboutView()
                .tabItem { Label("About", systemImage: "info.circle") }
        }
        .tint(LayaTheme.accent)
        .task { await engine.loadIfNeeded() }
    }
}

enum LayaTheme {
    static let ink = Color(red: 0.07, green: 0.09, blue: 0.11)
    static let panel = Color(red: 0.12, green: 0.14, blue: 0.16)
    static let accent = Color(red: 0.20, green: 0.85, blue: 0.72) // teal, not purple
    static let warn = Color(red: 0.95, green: 0.55, blue: 0.25)
    static let danger = Color(red: 0.95, green: 0.35, blue: 0.35)
    static let muted = Color.white.opacity(0.55)
}

enum Keyboard {
    static func dismiss() {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil
        )
    }
}

extension View {
    /// Done button above the keyboard + scroll-to-dismiss.
    func dismissesKeyboard() -> some View {
        self
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { Keyboard.dismiss() }
                        .fontWeight(.semibold)
                        .foregroundStyle(LayaTheme.accent)
                }
            }
    }
}
