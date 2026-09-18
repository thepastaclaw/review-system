// Print the on-screen windows of the calling user's GUI session (owner, layer, bounds).
// Used by the reviewqa verifier to prove that launched apps really get composited windows,
// which a wallpaper-only screencapture cannot show. Build: swiftc -O -o winlist winlist.swift
import CoreGraphics
import Foundation

let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else {
    print("CGWindowListCopyWindowInfo returned nil (no window server access)")
    exit(2)
}
var n = 0
for w in list {
    let owner = w["kCGWindowOwnerName"] as? String ?? "?"
    let layer = w["kCGWindowLayer"] as? Int ?? -1
    let name = w["kCGWindowName"] as? String ?? ""
    var rect = CGRect.zero
    if let b = w["kCGWindowBounds"] as? NSDictionary {
        CGRectMakeWithDictionaryRepresentation(b, &rect)
    }
    if layer != 0 { continue }  // normal app windows only
    n += 1
    print("\(owner)\t\(name)\t\(Int(rect.origin.x)),\(Int(rect.origin.y)) \(Int(rect.width))x\(Int(rect.height))")
}
print("windows(layer0)=\(n)")
