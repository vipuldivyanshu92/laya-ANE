import SwiftUI

@main
struct LayaANEApp: App {
    @StateObject private var engine = LayaEngine()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(engine)
                .preferredColorScheme(.dark)
        }
    }
}
