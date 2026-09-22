import SwiftUI

/// Laya-driven Snake — the fun counterpart to the laya-mlx terminal demo.
struct SnakeView: View {
    @EnvironmentObject var engine: LayaEngine
    @StateObject private var game = SnakeGameModel()
    @State private var lastDecision: String = "—"
    @State private var lastMs: Double = 0
    @State private var ticking = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                HStack {
                    StatChip(title: "Score", value: "\(game.score)")
                    StatChip(title: "Move", value: lastDecision)
                    StatChip(title: "Infer", value: String(format: "%.0fms", lastMs))
                }
                board
                HStack {
                    Button(ticking ? "Pause" : "Play") {
                        ticking.toggle()
                        if ticking { Task { await loop() } }
                    }
                    .buttonStyle(PrimaryButtonStyle())
                    Button("Reset") {
                        ticking = false
                        game.reset()
                        lastDecision = "—"
                    }
                    .buttonStyle(SecondaryButtonStyle())
                }
                Text("Each tick asks Laya for UP/DOWN/LEFT/RIGHT. Cycle-safety prefers legal moves when the model proposes a crash.")
                    .font(.caption)
                    .foregroundStyle(LayaTheme.muted)
                Spacer()
            }
            .padding(20)
            .background(LayaTheme.ink.ignoresSafeArea())
            .navigationTitle("Snake × Laya")
        }
    }

    private var board: some View {
        GeometryReader { geo in
            let cell = min(geo.size.width, geo.size.height) / CGFloat(game.size)
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(LayaTheme.panel)
                ForEach(0..<game.size, id: \.self) { y in
                    ForEach(0..<game.size, id: \.self) { x in
                        let pt = GridPos(x: x, y: y)
                        if game.body.contains(pt) {
                            Rectangle()
                                .fill(pt == game.body.first ? LayaTheme.accent : Color.white.opacity(0.75))
                                .frame(width: cell - 2, height: cell - 2)
                                .position(x: CGFloat(x) * cell + cell / 2, y: CGFloat(y) * cell + cell / 2)
                        } else if game.food == pt {
                            Circle()
                                .fill(LayaTheme.warn)
                                .frame(width: cell - 6, height: cell - 6)
                                .position(x: CGFloat(x) * cell + cell / 2, y: CGFloat(y) * cell + cell / 2)
                        }
                    }
                }
            }
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private func loop() async {
        while ticking && game.alive {
            await step()
            try? await Task.sleep(nanoseconds: 180_000_000)
        }
        ticking = false
    }

    private func step() async {
        let moves = game.candidateMoves()
        var descriptions: [String: String] = [:]
        for m in moves {
            if !m.legal { descriptions[m.dir.rawValue] = "Blocked. Collision." }
            else if m.dir == game.bestTowardFood() { descriptions[m.dir.rawValue] = "Safe. Best route to food." }
            else { descriptions[m.dir.rawValue] = "Safe. Slower route." }
        }
        let state = "Safe route: \(moves.contains{ $0.legal } ? "yes" : "no"). Food reachable through empty cells: yes."
        let q = Presets.snakeMove(descriptions: descriptions)
        let t0 = CFAbsoluteTimeGetCurrent()
        let report = await engine.predict(state: state, questions: [q])
        lastMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let probs = report.answers.first?.probabilities ?? [:]
        let proposed = probs.max(by: { $0.value < $1.value })?.key ?? "RIGHT"
        let legal = Set(moves.filter(\.legal).map(\.dir.rawValue))
        let executed = legal.contains(proposed)
            ? proposed
            : (legal.first ?? proposed)
        lastDecision = executed
        game.step(Direction(rawValue: executed) ?? .right)
    }
}

struct StatChip: View {
    let title: String
    let value: String
    var body: some View {
        VStack(spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(LayaTheme.muted)
            Text(value).font(.headline.monospaced()).foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity)
        .padding(10)
        .background(LayaTheme.panel)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

struct PrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .fontWeight(.semibold)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(LayaTheme.accent)
            .foregroundStyle(.black)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .opacity(configuration.isPressed ? 0.8 : 1)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .fontWeight(.semibold)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(LayaTheme.panel)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

struct GridPos: Hashable {
    var x: Int
    var y: Int
}

enum Direction: String {
    case up = "UP", down = "DOWN", left = "LEFT", right = "RIGHT"
    var delta: GridPos {
        switch self {
        case .up: return GridPos(x: 0, y: -1)
        case .down: return GridPos(x: 0, y: 1)
        case .left: return GridPos(x: -1, y: 0)
        case .right: return GridPos(x: 1, y: 0)
        }
    }
}

struct MoveOption {
    let dir: Direction
    let legal: Bool
}

@MainActor
final class SnakeGameModel: ObservableObject {
    let size = 12
    @Published var body: [GridPos] = []
    @Published var food = GridPos(x: 6, y: 6)
    @Published var score = 0
    @Published var alive = true
    private var dir: Direction = .right

    init() { reset() }

    func reset() {
        body = [GridPos(x: 3, y: 6), GridPos(x: 2, y: 6), GridPos(x: 1, y: 6)]
        food = GridPos(x: 9, y: 6)
        score = 0
        alive = true
        dir = .right
    }

    func candidateMoves() -> [MoveOption] {
        Direction.allCases.map { d in
            MoveOption(dir: d, legal: isLegal(d))
        }
    }

    func bestTowardFood() -> Direction {
        let head = body[0]
        let options = candidateMoves().filter(\.legal)
        return options.min(by: {
            manhattan(apply($0.dir, to: head), food) < manhattan(apply($1.dir, to: head), food)
        })?.dir ?? dir
    }

    func step(_ next: Direction) {
        guard alive else { return }
        if !isLegal(next) {
            alive = false
            return
        }
        dir = next
        let head = apply(next, to: body[0])
        body.insert(head, at: 0)
        if head == food {
            score += 1
            placeFood()
        } else {
            body.removeLast()
        }
    }

    private func isLegal(_ d: Direction) -> Bool {
        // No immediate reverse into neck
        if body.count > 1 {
            let neck = body[1]
            let nxt = apply(d, to: body[0])
            if nxt == neck { return false }
        }
        let nxt = apply(d, to: body[0])
        if nxt.x < 0 || nxt.y < 0 || nxt.x >= size || nxt.y >= size { return false }
        if body.dropLast().contains(nxt) { return false }
        return true
    }

    private func apply(_ d: Direction, to p: GridPos) -> GridPos {
        GridPos(x: p.x + d.delta.x, y: p.y + d.delta.y)
    }

    private func manhattan(_ a: GridPos, _ b: GridPos) -> Int {
        abs(a.x - b.x) + abs(a.y - b.y)
    }

    private func placeFood() {
        var open: [GridPos] = []
        for y in 0..<size {
            for x in 0..<size {
                let p = GridPos(x: x, y: y)
                if !body.contains(p) { open.append(p) }
            }
        }
        food = open.randomElement() ?? food
    }
}

extension Direction: CaseIterable {}
