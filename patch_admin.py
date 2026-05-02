import sys

with open('final/final/AdminDashboardView.swift', 'r') as f:
    content = f.read()

# We want to replace the import section to include Charts
import_section = """import SwiftUI
import Combine

#if os(macOS)"""

new_import_section = """import SwiftUI
import Combine
import Charts

#if os(macOS)"""

content = content.replace(import_section, new_import_section)

# Now we replace the AdminDashboardView struct up to the #else
start_marker = "struct AdminDashboardView: View {"
end_marker = "#else"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers")
    sys.exit(1)

new_view = """struct EnergyDataPoint: Identifiable {
    let id = UUID()
    let time: String
    let value: Double
}

let mockGraphData: [EnergyDataPoint] = [
    .init(time: "08:00", value: 12),
    .init(time: "10:00", value: 45),
    .init(time: "12:00", value: 68),
    .init(time: "14:00", value: 120),
    .init(time: "16:00", value: 240),
    .init(time: "18:00", value: 310),
    .init(time: "20:00", value: 428.5)
]

struct AdminDashboardView: View {
    @StateObject private var viewModel = AdminDashboardViewModel()
    
    var body: some View {
        ZStack {
            Color.white.ignoresSafeArea()
            
            VStack(spacing: 0) {
                headerBar
                
                Divider().opacity(0.5)
                
                HStack(spacing: 0) {
                    // LEFT SIDEBAR (Stats)
                    ScrollView {
                        VStack(alignment: .leading, spacing: 48) {
                            energyGraphSection
                            statsSection
                            lastDetectionSection
                        }
                        .padding(40)
                    }
                    .frame(width: 320)
                    
                    Divider().opacity(0.5)
                    
                    // MIDDLE (Live Feed)
                    VStack {
                        liveFeedSection
                    }
                    .frame(maxWidth: .infinity)
                    
                    Divider().opacity(0.5)
                    
                    // RIGHT SIDEBAR (Users & Logs)
                    VStack(spacing: 0) {
                        usersSection
                        Divider().opacity(0.5)
                        logsSection
                    }
                    .frame(width: 320)
                }
            }
        }
        .frame(minWidth: 1080, minHeight: 720)
    }
    
    private var headerBar: some View {
        HStack(alignment: .center) {
            Text("Ethereal Charge.")
                .font(MacTypography.serif(28))
                .foregroundColor(.black)
            
            Spacer()
            
            HStack(spacing: 16) {
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.isConnected ? Color.green : Color.gray.opacity(0.3))
                        .frame(width: 6, height: 6)
                    Text(viewModel.isConnected ? "Connected" : "Disconnected")
                        .font(MacTypography.sans(12))
                        .foregroundColor(.gray)
                }
                
                Divider().frame(height: 16)
                
                TextField("IP", text: $viewModel.serverIP)
                    .font(MacTypography.sans(13))
                    .textFieldStyle(.plain)
                    .frame(width: 100)
                    .foregroundColor(.black)
                
                Text(":")
                    .foregroundColor(.gray)
                
                TextField("Port", text: $viewModel.serverPort)
                    .font(MacTypography.sans(13))
                    .textFieldStyle(.plain)
                    .frame(width: 40)
                    .foregroundColor(.black)
                
                Button(action: {
                    if viewModel.isConnected {
                        viewModel.disconnect()
                    } else {
                        viewModel.connect()
                    }
                }) {
                    Text(viewModel.isConnected ? "Disconnect" : "Connect")
                        .font(MacTypography.sans(12))
                        .foregroundColor(.white)
                        .padding(.vertical, 6)
                        .padding(.horizontal, 14)
                        .background(Color.black, in: Capsule())
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.03), in: Capsule())
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 20)
    }
    
    private var energyGraphSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Energy Delivered")
                .font(MacTypography.sans(12))
                .foregroundColor(.gray)
                .textCase(.uppercase)
                .tracking(1.0)
            
            Text(String(format: "%.1f kWh", viewModel.totalEnergyKWh))
                .font(MacTypography.serif(48))
                .foregroundColor(.black)
            
            Chart(mockGraphData) { item in
                AreaMark(
                    x: .value("Time", item.time),
                    y: .value("Energy", item.value)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [Color.black.opacity(0.05), Color.clear],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                
                LineMark(
                    x: .value("Time", item.time),
                    y: .value("Energy", item.value)
                )
                .foregroundStyle(Color.black)
                .lineStyle(StrokeStyle(lineWidth: 1.5))
            }
            .frame(height: 120)
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
        }
    }
    
    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(spacing: 40) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Total Scans")
                        .font(MacTypography.sans(12))
                        .foregroundColor(.gray)
                        .textCase(.uppercase)
                        .tracking(1.0)
                    Text("\\(viewModel.totalScans)")
                        .font(MacTypography.serif(32))
                        .foregroundColor(.black)
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Registered")
                        .font(MacTypography.sans(12))
                        .foregroundColor(.gray)
                        .textCase(.uppercase)
                        .tracking(1.0)
                    Text("\\(viewModel.registeredHits)")
                        .font(MacTypography.serif(32))
                        .foregroundColor(.black)
                }
            }
        }
    }
    
    private var lastDetectionSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Latest Detection")
                .font(MacTypography.sans(12))
                .foregroundColor(.gray)
                .textCase(.uppercase)
                .tracking(1.0)
                
            if !viewModel.lastPlate.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text(viewModel.lastPlate)
                        .font(.system(size: 28, weight: .regular, design: .monospaced))
                        .foregroundColor(.black)
                    
                    if !viewModel.lastOwner.isEmpty {
                        Text("\\(viewModel.lastOwner) • \\(viewModel.lastBalance)")
                            .font(MacTypography.sans(14))
                            .foregroundColor(.gray)
                    } else {
                        Text("Unregistered")
                            .font(MacTypography.sans(14))
                            .foregroundColor(.red)
                    }
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.black.opacity(0.03), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            } else {
                Text("Waiting for vehicle...")
                    .font(MacTypography.sans(14))
                    .foregroundColor(.gray)
            }
        }
    }
    
    private var liveFeedSection: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Live Camera Feed")
                    .font(MacTypography.serif(24))
                    .foregroundColor(.black)
                Spacer()
                if viewModel.feedImage != nil {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 6, height: 6)
                        Text("LIVE")
                            .font(MacTypography.sans(11))
                            .tracking(1.0)
                            .foregroundColor(.black)
                    }
                }
            }
            .padding(40)
            
            Spacer()
            
            ZStack {
                if let img = viewModel.feedImage {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                } else {
                    Text(viewModel.isConnected ? "Awaiting visual feed..." : "Camera offline")
                        .font(MacTypography.sans(14))
                        .foregroundColor(.gray)
                }
            }
            .padding(.horizontal, 40)
            .padding(.bottom, 40)
            
            Spacer()
        }
    }
    
    private var usersSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("User Directory")
                    .font(MacTypography.serif(20))
                    .foregroundColor(.black)
                Spacer()
                Text("\\(viewModel.users.count)")
                    .font(MacTypography.sans(12))
                    .foregroundColor(.gray)
            }
            .padding(24)
            
            List(viewModel.users) { user in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(user.owner)
                            .font(MacTypography.sans(14))
                            .foregroundColor(.black)
                        Text(user.plate)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.gray)
                    }
                    Spacer()
                    Text(user.balance)
                        .font(MacTypography.sans(14))
                        .foregroundColor(.black)
                }
                .padding(.vertical, 8)
                .listRowSeparator(.visible)
                .listRowSeparatorTint(Color.black.opacity(0.1))
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
        }
        .frame(maxHeight: .infinity)
    }
    
    private var logsSection: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Terminal Logs")
                    .font(MacTypography.serif(20))
                    .foregroundColor(.black)
                Spacer()
            }
            .padding(24)
            
            ScrollViewReader { scrollProxy in
                ScrollView {
                    Text(viewModel.logText)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.black.opacity(0.6))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 24)
                        .padding(.bottom, 24)
                        .id("LogEnd")
                }
                .onChange(of: viewModel.logEntries.count) { _ in
                    withAnimation(.easeOut(duration: 0.2)) {
                        scrollProxy.scrollTo("LogEnd", anchor: .bottom)
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }
}
"""

new_content = content[:start_idx] + new_view + "\n" + content[end_idx:]

with open('final/final/AdminDashboardView.swift', 'w') as f:
    f.write(new_content)

print("Patch successful!")
